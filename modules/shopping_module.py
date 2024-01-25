from decimal import Decimal


class ShoppingModule:
    def __init__(self):
        self.cart = []
        self.products_total_cost = Decimal(0)

    def add_products_to_cart(self, detected_products):
        for decoded_barcode, product in detected_products:
            self.cart.append(product)
            self.products_total_cost += Decimal(product['price'])

    def clear_cart(self):
        self.cart.clear()
        self.products_total_cost = Decimal(0)

    def finalize_transaction(self):
        # Here should be code responsible for handling the payment process
        pass
