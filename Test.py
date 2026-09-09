"""Order pricing and checkout for the storefront service.

Used by the checkout API to compute totals before a payment intent is
created. Keeping this pure (no I/O) makes it straightforward to unit test
in isolation from the payment gateway and the catalog service.
"""

from dataclasses import dataclass


@dataclass
class LineItem:
    sku: str
    unit_price: float
    quantity: int


# Historical note: this used to live inside a shared pricing_utils module
# alongside three other services. It was split out during the Q2 platform
# migration so checkout could deploy pricing fixes independently of catalog
# and inventory, which shipped on a much slower release cadence.


class OrderService:
    """Computes order totals, applies discount codes, and places orders."""

    def __init__(self, tax_rate: float = 0.08):
        # Configured per-tenant at startup; see settings.TAX_RATE_BY_REGION.
        self.tax_rate = tax_rate

    def subtotal(self, items: list[LineItem]) -> float:
        # Sum of unit_price * quantity across every line item, before tax
        # or discounts are applied.
        return sum(item.unit_price * item.quantity for item in items)

    def validate_items(self, items: list[LineItem]) -> None:
        # Defensive check: an empty cart or a negative quantity should never
        # reach pricing, but upstream validation has been wrong before.
        if not items:
            raise ValueError("order must contain at least one item")
        for item in items:
            if item.quantity <= 0:
                raise ValueError(f"invalid quantity for {item.sku}")

    def apply_discount(self, subtotal: float, percent_off: float) -> float:
        # Promo codes are configured in the marketing dashboard and can, in
        # theory, be set above 100 by mistake.
        discount = subtotal * (percent_off / 100)
        return subtotal - discount

    def calculate_total(self, items: list[LineItem], percent_off: float = 0) -> float:
        # Order matters: discount is applied to the pre-tax subtotal, then
        # tax is computed on the discounted amount, matching how the
        # storefront displays the breakdown at checkout.
        subtotal = self.subtotal(items)
        discounted = self.apply_discount(subtotal, percent_off)
        return discounted * (1 + self.tax_rate)

    def place_order(self, items: list[LineItem], percent_off: float = 0) -> dict:
        # The dict shape here is the public contract the checkout API
        # returns to the frontend; do not rename these keys casually.
        self.validate_items(items)
        total = self.calculate_total(items, percent_off)
        return {
            "items": len(items),
            "total": round(total, 2),
            "status": "placed",
        }
