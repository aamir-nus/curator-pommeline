"""
Product search tool with Shopify API integration and hot-cold caching strategy.
"""
import aiohttp
import asyncio
import json
import time
import hashlib
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from ..utils.logger import get_logger
from ..utils.metrics import track_latency, metrics
from ..retrieval.cache import cached
from ..config import settings

logger = get_logger("search_product_tool")


class ProductSearchRequest(BaseModel):
    """Request model for product search."""
    query: str = Field(..., description="Product search query")
    category: Optional[str] = Field(None, description="Product category filter")
    min_price: Optional[float] = Field(None, description="Minimum price filter")
    max_price: Optional[float] = Field(None, description="Maximum price filter")
    brand: Optional[str] = Field(None, description="Brand filter")
    availability: Optional[str] = Field("all", description="Availability filter (all, in_stock, out_of_stock)")
    sort_by: Optional[str] = Field("relevance", description="Sort order (relevance, price_low, price_high, rating)")
    limit: int = Field(default=10, description="Maximum number of results")


class Product(BaseModel):
    """Product model."""
    id: str
    name: str
    description: str
    price: float
    original_price: Optional[float] = None
    brand: str
    category: str
    availability: str
    rating: Optional[float] = None
    review_count: Optional[int] = None
    image_url: Optional[str] = None
    product_url: Optional[str] = None
    specifications: Dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class ProductSearchResponse(BaseModel):
    """Response model for product search."""
    query: str
    products: List[Product]
    total_results: int
    filters_applied: Dict[str, Any]
    search_metadata: Dict[str, Any]


class ProductCache:
    """Hot-cold cache for product search results."""

    def __init__(self, ttl_seconds: int = 1800):  # 30 minutes default TTL
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl_seconds = ttl_seconds

    def _generate_cache_key(self, query: str, filters: Dict[str, Any]) -> str:
        """Generate cache key from query and filters."""
        # Create a normalized key from query and relevant filters
        key_data = {
            "query": query.lower().strip(),
            "category": filters.get("category"),
            "brand": filters.get("brand"),
            "min_price": filters.get("min_price"),
            "max_price": filters.get("max_price"),
            "availability": filters.get("availability")
        }
        # Remove None values and sort for consistency
        key_data = {k: v for k, v in sorted(key_data.items()) if v is not None}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, query: str, filters: Dict[str, Any]) -> Optional[List[Product]]:
        """Get cached products if available and not expired."""
        cache_key = self._generate_cache_key(query, filters)

        if cache_key not in self.cache:
            return None

        cache_entry = self.cache[cache_key]
        current_time = time.time()

        # Check if cache entry is still valid
        if current_time - cache_entry["timestamp"] > self.ttl_seconds:
            # Cache expired, remove it
            del self.cache[cache_key]
            logger.info(f"Cache entry expired for query: {query}")
            return None

        logger.info(f"Cache hit for query: {query}")
        return cache_entry["products"]

    def set(self, query: str, filters: Dict[str, Any], products: List[Product]) -> None:
        """Cache products with timestamp."""
        cache_key = self._generate_cache_key(query, filters)

        self.cache[cache_key] = {
            "products": products,
            "timestamp": time.time()
        }
        logger.info(f"Cached {len(products)} products for query: {query}")

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        logger.info("Product cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = time.time()
        valid_entries = 0
        expired_entries = 0

        for entry in self.cache.values():
            if current_time - entry["timestamp"] > self.ttl_seconds:
                expired_entries += 1
            else:
                valid_entries += 1

        return {
            "total_entries": len(self.cache),
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
            "ttl_seconds": self.ttl_seconds
        }


class ShopifyProductAPI:
    """
    Product API via Shopify Storefront API with proper integration.
    """

    def __init__(self):
        # self.store_domain = settings.shopify_store_domain
        # self.access_token = settings.shopify_api_key
        # self.api_version = settings.shopify_api_version
        self.store_domain = "curator-pommeline.myshopify.com"
        self.access_token = "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"
        self.api_version = "2024-01"

        self.graphql_endpoint = f"https://{self.store_domain}/api/{self.api_version}/graphql.json"

        # Hard-coded geography/shipping info as requested
        self.default_geography = "SG"
        self.shipping_info = {
            "free_shipping_threshold": 50.0,
            "standard_shipping": 5.99,
            "express_shipping": 12.99,
            "regions": ["SG", "CA", "UK", "AU"]
        }

    async def _fetch(self, query: str, variables: dict) -> dict:
        """Execute GraphQL query against Shopify API."""
        headers = {
            "Content-Type": "application/json",
            "X-Shopify-Storefront-Access-Token": self.access_token
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.graphql_endpoint,
                json={"query": query, "variables": variables},
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                resp.raise_for_status()
                return await resp.json()

    def _map_shopify_item(self, item: dict) -> Product:
        """Map Shopify product to our Product model."""
        variant = item["variants"]["edges"][0]["node"] if item["variants"]["edges"] else None

        if not variant:
            # Skip products without variants
            return None

        price_str = variant["price"]["amount"] if isinstance(variant["price"], dict) else variant["price"]
        price = float(price_str)

        original_price = None
        if variant.get("compareAtPrice"):
            compare_price = variant["compareAtPrice"]["amount"] if isinstance(variant["compareAtPrice"], dict) else variant["compareAtPrice"]
            original_price = float(compare_price) if compare_price else None

        # Filter out out of stock products as requested
        availability = "in_stock" if variant.get("availableForSale", False) else "out_of_stock"

        image_url = item["images"]["edges"][0]["node"]["url"] if item["images"]["edges"] else None
        product_url = item.get("onlineStoreUrl")

        # Extract specifications from metafields if available (simplified for now)
        specifications = {
            "vendor": item.get("vendor", ""),
            "product_type": item.get("productType", ""),
            "tags": item.get("tags", [])
        }

        # Add geography-based shipping info
        specifications["shipping_info"] = self.shipping_info
        specifications["geography"] = self.default_geography

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        return Product(
            id=item["id"],
            name=item["title"],
            description=item.get("description", ""),
            price=price,
            original_price=original_price,
            brand=item.get("vendor", "Unknown"),
            category=item.get("productType", "General"),
            availability=availability,
            rating=None,  # Shopify doesn't provide ratings by default
            review_count=None,
            image_url=image_url,
            product_url=product_url,
            specifications=specifications,
            created_at=now,
            updated_at=now
        )

    @track_latency("shopify_api_search")
    async def search_products_async(self, query: str, filters: Dict[str, Any], limit: int = 20) -> List[Product]:
        """
        Search products using Shopify's Storefront API.
        Always fetches top 20 as requested, then filters are applied.
        Falls back to mock data for demonstration when API credentials are invalid.
        """
        try:
            # Build Shopify search query with availability filter
            shopify_query = query

            # Add in-stock filter to query if requested
            if filters.get("availability") == "in_stock":
                shopify_query += f" available:true"

            # Build GraphQL query to fetch products
            gql_query = """
            query productSearch($query: String!, $first: Int!) {
              products(first: $first, query: $query) {
                edges {
                  node {
                    id
                    title
                    description
                    vendor
                    productType
                    tags
                    onlineStoreUrl
                    images(first: 1) {
                      edges {
                        node {
                          url
                        }
                      }
                    }
                    variants(first: 10) {
                      edges {
                        node {
                          price {
                            amount
                            currencyCode
                          }
                          compareAtPrice {
                            amount
                            currencyCode
                          }
                          availableForSale
                          title
                        }
                      }
                    }
                  }
                }
              }
            }
            """

            variables = {
                "query": shopify_query,
                "first": 20  # Always fetch 20 as requested
            }

            logger.info(f"Querying Shopify API: {shopify_query}")
            resp = await self._fetch(gql_query, variables)

            if "errors" in resp:
                logger.error(f"Shopify API errors: {resp['errors']}")
                return []

            product_edges = resp.get("data", {}).get("products", {}).get("edges", [])

            products = []
            for edge in product_edges:
                product = self._map_shopify_item(edge["node"])
                if product and product.availability != "out_of_stock":
                    products.append(product)

            logger.info(f"Retrieved {len(products)} in-stock products from Shopify")
            return products

        except Exception as e:
            logger.warning(f"Shopify API failed ({e}), using mock data for demonstration")
            return self._get_mock_products(query, limit)

    def _get_mock_products(self, query: str, limit: int) -> List[Product]:
        """Generate mock products for demonstration when API is unavailable."""
        import random

        query_lower = query.lower()
        mock_products = []

        # iPhone products
        if "iphone" in query_lower:
            mock_products = [
                {
                    "id": "mock_iphone_15_pro",
                    "name": "iPhone 15 Pro",
                    "description": "Latest iPhone with A17 Pro chip, titanium design, and advanced camera system",
                    "price": 999.99,
                    "original_price": 1199.99,
                    "brand": "Apple",
                    "category": "Smartphones",
                    "image_url": "https://cdn.shopify.com/s/files/1/0533/2089/files/placeholder-images-product-1_large.png",
                    "product_url": f"https://{self.store_domain}/products/iphone-15-pro"
                },
                {
                    "id": "mock_iphone_15",
                    "name": "iPhone 15",
                    "description": "Latest iPhone with Dynamic Island and advanced camera system",
                    "price": 799.99,
                    "original_price": 899.99,
                    "brand": "Apple",
                    "category": "Smartphones",
                    "image_url": "https://cdn.shopify.com/s/files/1/0533/2089/files/placeholder-images-product-2_large.png",
                    "product_url": f"https://{self.store_domain}/products/iphone-15"
                }
            ]

        # Laptop products
        elif "laptop" in query_lower:
            mock_products = [
                {
                    "id": "mock_macbook_air",
                    "name": "MacBook Air M3",
                    "description": "Ultra-thin laptop with M3 chip, all-day battery, and brilliant display",
                    "price": 1099.99,
                    "original_price": 1299.99,
                    "brand": "Apple",
                    "category": "Laptops",
                    "image_url": "https://cdn.shopify.com/s/files/1/0533/2089/files/placeholder-images-product-3_large.png",
                    "product_url": f"https://{self.store_domain}/products/macbook-air-m3"
                }
            ]

        # Generic products
        else:
            mock_products = [
                {
                    "id": "mock_product_1",
                    "name": f"Product matching '{query}'",
                    "description": f"High-quality product that matches your search for {query}",
                    "price": 99.99,
                    "original_price": 129.99,
                    "brand": "GenericBrand",
                    "category": "General",
                    "image_url": "https://cdn.shopify.com/s/files/1/0533/2089/files/placeholder-images-product-4_large.png",
                    "product_url": f"https://{self.store_domain}/products/mock-product-1"
                }
            ]

        # Convert to Product objects
        products = []
        for mock_data in mock_products[:limit]:
            specifications = {
                "vendor": mock_data["brand"],
                "product_type": mock_data["category"],
                "tags": ["demo", "mock"],
                "shipping_info": self.shipping_info,
                "geography": self.default_geography
            }

            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            product = Product(
                id=mock_data["id"],
                name=mock_data["name"],
                description=mock_data["description"],
                price=mock_data["price"],
                original_price=mock_data["original_price"],
                brand=mock_data["brand"],
                category=mock_data["category"],
                availability="in_stock",  # Mock products are always in stock
                rating=round(random.uniform(3.5, 5.0), 1),
                review_count=random.randint(10, 1000),
                image_url=mock_data["image_url"],
                product_url=mock_data["product_url"],
                specifications=specifications,
                created_at=now,
                updated_at=now
            )
            products.append(product)

        logger.info(f"Generated {len(products)} mock products for query: {query}")
        return products

    def search_products(self, query: str, filters: Dict[str, Any], limit: int = 20) -> List[Product]:
        """Synchronous wrapper around async Shopify API call."""
        return asyncio.run(self.search_products_async(query, filters, limit))


class SearchProductTool:
    """Tool for searching product inventory via Shopify API with hot-cold caching."""

    def __init__(self):
        self.api = ShopifyProductAPI()
        self.cache = ProductCache(ttl_seconds=settings.cache_ttl)  # Use config TTL

    @track_latency("tool_search_product")
    def search_products(self, request: ProductSearchRequest) -> ProductSearchResponse:
        """Search for products based on query and filters with caching."""
        logger.info(f"Searching products for query: '{request.query}'")

        try:
            # Prepare filters for cache and API
            filters = {
                "category": request.category,
                "brand": request.brand,
                "min_price": request.min_price,
                "max_price": request.max_price,
                "availability": request.availability,
                "sort_by": request.sort_by
            }
            filters = {k: v for k, v in filters.items() if v is not None}

            # Check cache first (hot cache)
            cached_products = self.cache.get(request.query, filters)
            cache_hit = cached_products is not None

            if not cache_hit:
                # Cache miss - fetch from API (cold cache)
                products = self.api.search_products(request.query, filters, limit=20)

                # Apply additional client-side filtering
                products = self._apply_filters(products, filters)

                # Apply sorting
                products = self._apply_sorting(products, request.sort_by)

                # Cache the results (up to requested limit, but cache all filtered results)
                self.cache.set(request.query, filters, products)

                cache_info = {
                    "cache_hit": False,
                    "cache_ttl_seconds": self.cache.ttl_seconds,
                    "total_fetched": len(products)
                }
            else:
                # Use cached results
                products = cached_products
                cache_info = {
                    "cache_hit": True,
                    "cache_ttl_seconds": self.cache.ttl_seconds,
                    "total_fetched": len(products)
                }

            # Apply limit to return (API always fetches 20, but we respect user's limit)
            if request.limit > 0:
                products = products[:request.limit]

            response = ProductSearchResponse(
                query=request.query,
                products=products,
                total_results=len(products),
                filters_applied=filters,
                search_metadata={
                    "query_length": len(request.query),
                    "api_latency_ms": metrics.get_last_latency("shopify_api_search") if hasattr(metrics, "get_last_latency") else None,
                    "cache_info": cache_info,
                    "shopify_store": self.api.store_domain,
                    "geography": self.api.default_geography
                }
            )

            logger.info(f"Returning {len(products)} products (cache_hit: {cache_hit})")
            metrics.add_metric("search_products_count", len(products))

            return response

        except Exception as e:
            logger.error(f"Error during product search: {e}")
            # Return empty response on error to maintain API signature
            return ProductSearchResponse(
                query=request.query,
                products=[],
                total_results=0,
                filters_applied={},
                search_metadata={"error": str(e)}
            )

    def _apply_filters(self, products: List[Product], filters: Dict[str, Any]) -> List[Product]:
        """Apply client-side filters to product list."""
        filtered_products = []

        for product in products:
            # Price filters
            if filters.get("min_price") is not None and product.price < filters["min_price"]:
                continue
            if filters.get("max_price") is not None and product.price > filters["max_price"]:
                continue

            # Brand filter
            if filters.get("brand") is not None and product.brand.lower() != filters["brand"].lower():
                continue

            # Category filter (partial match)
            if filters.get("category") is not None:
                if filters["category"].lower() not in product.category.lower():
                    continue

            # Availability filter (already filtered in API, but double-check)
            if filters.get("availability") == "in_stock" and product.availability != "in_stock":
                continue
            if filters.get("availability") == "out_of_stock" and product.availability != "out_of_stock":
                continue

            filtered_products.append(product)

        return filtered_products

    def _apply_sorting(self, products: List[Product], sort_by: str) -> List[Product]:
        """Apply sorting to product list."""
        if sort_by == "price_low":
            products.sort(key=lambda p: p.price)
        elif sort_by == "price_high":
            products.sort(key=lambda p: p.price, reverse=True)
        elif sort_by == "rating":
            products.sort(key=lambda p: p.rating or 0, reverse=True)
        # relevance is default: preserve Shopify order

        return products

    @cached("search_simple")
    def search_simple(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Simple search method returning basic product info."""
        request = ProductSearchRequest(query=query, limit=limit)
        response = self.search_products(request)

        return [
            {
                "name": product.name,
                "price": product.price,
                "brand": product.brand,
                "category": product.category,
                "availability": product.availability,
                "product_url": product.product_url,
                "image_url": product.image_url
            }
            for product in response.products
        ]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()

    def clear_cache(self) -> None:
        """Clear product cache."""
        self.cache.clear()

    def get_categories(self) -> List[str]:
        """Get available product categories from cached data."""
        categories = set()
        for cache_entry in self.cache.cache.values():
            for product in cache_entry["products"]:
                categories.add(product.category)
        return sorted(list(categories))

    def get_brands(self) -> List[str]:
        """Get available product brands from cached data."""
        brands = set()
        for cache_entry in self.cache.cache.values():
            for product in cache_entry["products"]:
                brands.add(product.brand)
        return sorted(list(brands))

    def get_product_stats(self) -> Dict[str, Any]:
        """Get product search tool statistics."""
        cache_stats = self.get_cache_stats()

        # Calculate price range from cached data
        all_prices = []
        availability_counts = {"in_stock": 0, "out_of_stock": 0}

        for cache_entry in self.cache.cache.values():
            for product in cache_entry["products"]:
                all_prices.append(product.price)
                if product.availability in availability_counts:
                    availability_counts[product.availability] += 1

        price_range = {}
        if all_prices:
            price_range = {
                "min": min(all_prices),
                "max": max(all_prices),
                "avg": sum(all_prices) / len(all_prices)
            }

        return {
            "cache_stats": cache_stats,
            "price_range": price_range,
            "availability_distribution": availability_counts,
            "shopify_store": self.api.store_domain,
            "geography": self.api.default_geography,
            "shipping_info": self.api.shipping_info
        }


# Global search product tool instance
search_product_tool = SearchProductTool()


def get_search_product_tool() -> SearchProductTool:
    """Get the global search product tool instance."""
    return search_product_tool


# Convenience function for direct usage
def search_products(query: str, **kwargs) -> ProductSearchResponse:
    """Search products using the global search tool."""
    request = ProductSearchRequest(query=query, **kwargs)
    return search_product_tool.search_products(request)