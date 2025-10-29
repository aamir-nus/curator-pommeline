"""
Product search tool with mocked inventory API.
"""

import json
import time
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field

from ..utils.logger import get_logger
from ..utils.metrics import track_latency, metrics
from ..retrieval.cache import cached

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


class MockProductAPI:
    """Mock product inventory API for PoC."""

    def __init__(self):
        self.products = self._generate_mock_products()
        self.last_update = time.time()

    def _generate_mock_products(self) -> List[Product]:
        """Generate mock product data."""
        products = []

        # Mock product templates
        product_templates = [
            {
                "name": "iPhone 16 Pro",
                "description": "Latest iPhone with A18 Pro chip, titanium design, and advanced camera system",
                "price": 999.99,
                "brand": "Apple",
                "category": "Smartphones",
                "specifications": {
                    "screen_size": "6.3 inches",
                    "storage": "256GB",
                    "camera": "48MP main camera",
                    "processor": "A18 Pro",
                    "battery": "All-day battery life"
                }
            },
            {
                "name": "Samsung Galaxy S24 Ultra",
                "description": "Premium Android phone with S Pen, incredible zoom, and AI features",
                "price": 1199.99,
                "brand": "Samsung",
                "category": "Smartphones",
                "specifications": {
                    "screen_size": "6.8 inches",
                    "storage": "512GB",
                    "camera": "200MP main camera",
                    "processor": "Snapdragon 8 Gen 3",
                    "battery": "5000mAh"
                }
            },
            {
                "name": "AirPods Pro 2",
                "description": "Active noise cancellation, spatial audio, and personalized fit",
                "price": 249.99,
                "brand": "Apple",
                "category": "Audio",
                "specifications": {
                    "noise_cancellation": "Active Noise Cancellation",
                    "battery_life": "6 hours listening time",
                    "charging_case": "MagSafe charging case",
                    "connectivity": "Bluetooth 5.3"
                }
            },
            {
                "name": "Sony WH-1000XM5",
                "description": "Industry-leading noise canceling headphones with exceptional sound quality",
                "price": 399.99,
                "brand": "Sony",
                "category": "Audio",
                "specifications": {
                    "noise_cancellation": "Industry-leading ANC",
                    "battery_life": "30 hours",
                    "drivers": "30mm drivers",
                    "connectivity": "Bluetooth 5.2"
                }
            },
            {
                "name": "MacBook Air M3",
                "description": "Ultra-thin laptop with M3 chip, all-day battery, and brilliant display",
                "price": 1099.99,
                "brand": "Apple",
                "category": "Laptops",
                "specifications": {
                    "processor": "Apple M3 chip",
                    "memory": "8GB unified memory",
                    "storage": "256GB SSD",
                    "display": "13.6-inch Liquid Retina",
                    "battery": "Up to 18 hours"
                }
            },
            {
                "name": "Dell XPS 13",
                "description": "Compact Windows laptop with premium build and powerful performance",
                "price": 899.99,
                "brand": "Dell",
                "category": "Laptops",
                "specifications": {
                    "processor": "Intel Core i5-1335U",
                    "memory": "16GB DDR5",
                    "storage": "512GB NVMe SSD",
                    "display": "13.4-inch FHD+",
                    "battery": "Up to 12 hours"
                }
            }
        ]

        # Generate variations
        for i, template in enumerate(product_templates):
            for j in range(3):  # Create 3 variations per template
                product = Product(
                    id=f"product_{i}_{j}",
                    name=template["name"],
                    description=template["description"],
                    price=template["price"] * (1 + random.uniform(-0.1, 0.1)),  # Price variation
                    original_price=template["price"] * 1.2 if random.random() > 0.5 else None,
                    brand=template["brand"],
                    category=template["category"],
                    availability=random.choice(["in_stock", "in_stock", "in_stock", "out_of_stock"]),
                    rating=round(random.uniform(3.5, 5.0), 1),
                    review_count=random.randint(10, 1000),
                    image_url=f"https://example.com/images/product_{i}_{j}.jpg",
                    product_url=f"https://example.com/products/product_{i}_{j}",
                    specifications=template["specifications"],
                    created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - random.randint(1, 365) * 24 * 3600)),
                    updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                )
                products.append(product)

        return products

    @track_latency("mock_api_search")
    def search_products(self, query: str, filters: Dict[str, Any]) -> List[Product]:
        """Mock API search with realistic delay."""
        # Simulate API latency
        time.sleep(random.uniform(0.1, 0.3))

        # Filter products based on query and filters
        filtered_products = []
        query_lower = query.lower()

        for product in self.products:
            # Text search
            text_match = (
                query_lower in product.name.lower() or
                query_lower in product.description.lower() or
                query_lower in product.brand.lower() or
                query_lower in product.category.lower()
            )

            if not text_match:
                continue

            # Apply filters
            if filters.get("category") and product.category != filters["category"]:
                continue
            if filters.get("brand") and product.brand != filters["brand"]:
                continue
            if filters.get("min_price") and product.price < filters["min_price"]:
                continue
            if filters.get("max_price") and product.price > filters["max_price"]:
                continue
            if filters.get("availability") and filters["availability"] != "all":
                if product.availability != filters["availability"]:
                    continue

            filtered_products.append(product)

        # Sort results
        sort_by = filters.get("sort_by", "relevance")
        if sort_by == "price_low":
            filtered_products.sort(key=lambda p: p.price)
        elif sort_by == "price_high":
            filtered_products.sort(key=lambda p: p.price, reverse=True)
        elif sort_by == "rating":
            filtered_products.sort(key=lambda p: p.rating or 0, reverse=True)
        # relevance is the default (no additional sorting needed)

        return filtered_products


class SearchProductTool:
    """Tool for searching product inventory."""

    def __init__(self):
        self.mock_api = MockProductAPI()

    @track_latency("tool_search_product")
    def search_products(self, request: ProductSearchRequest) -> ProductSearchResponse:
        """Search for products based on query and filters."""
        logger.info(f"Searching products for query: '{request.query}'")

        try:
            # Prepare filters
            filters = {
                "category": request.category,
                "brand": request.brand,
                "min_price": request.min_price,
                "max_price": request.max_price,
                "availability": request.availability,
                "sort_by": request.sort_by
            }

            # Remove None values
            filters = {k: v for k, v in filters.items() if v is not None}

            # Search products
            products = self.mock_api.search_products(request.query, filters)

            # Apply limit
            if request.limit > 0:
                products = products[:request.limit]

            # Create response
            response = ProductSearchResponse(
                query=request.query,
                products=products,
                total_results=len(products),
                filters_applied=filters,
                search_metadata={
                    "query_length": len(request.query),
                    "api_latency_ms": random.uniform(100, 300),  # Mock API latency
                    "cache_hit": False,  # Would be True if cached
                    "total_available": len(self.mock_api.products)
                }
            )

            logger.info(f"Found {len(products)} products")
            metrics.add_metric("search_products_count", len(products))

            return response

        except Exception as e:
            logger.error(f"Error during product search: {e}")
            raise

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
                "availability": product.availability
            }
            for product in response.products
        ]

    def get_categories(self) -> List[str]:
        """Get available product categories."""
        return list(set(product.category for product in self.mock_api.products))

    def get_brands(self) -> List[str]:
        """Get available product brands."""
        return list(set(product.brand for product in self.mock_api.products))

    def get_product_stats(self) -> Dict[str, Any]:
        """Get product search tool statistics."""
        products = self.mock_api.products
        return {
            "total_products": len(products),
            "categories": self.get_categories(),
            "brands": self.get_brands(),
            "availability_distribution": {
                status: len([p for p in products if p.availability == status])
                for status in set(p.availability for p in products)
            },
            "price_range": {
                "min": min(p.price for p in products),
                "max": max(p.price for p in products),
                "avg": sum(p.price for p in products) / len(products)
            }
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