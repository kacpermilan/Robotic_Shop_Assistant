from decimal import Decimal


class ShoppingModule:
    """
    A class to manage the shopping cart and transactions for a shopping assistant.

    Attributes:
    -----------
    cart : list
        A list to hold the products added to the shopping cart.
    products_total_cost : Decimal
        The total cost of the products in the cart.

    Methods:
    --------
    add_products_to_cart(self, detected_products):
        Adds a list of detected products to the shopping cart and updates the total cost.

    clear_cart(self):
        Clears all items from the cart and resets the total cost to zero.

    finalize_transaction(self):
        Finalizes the transaction, intended for handling the payment process.
    """
    def __init__(self):
        self.cart = []
        self.products_total_cost = Decimal(0)

    def add_products_to_cart(self, detected_products):
        """
        Adds detected products to the cart and updates the total cost.
        """
        for decoded_barcode, product in detected_products:
            self.cart.append(product)
            self.products_total_cost += Decimal(product['price'])

    def clear_cart(self):
        """
        Clears the cart of all products and resets the total cost.
        """
        self.cart.clear()
        self.products_total_cost = Decimal(0)

    def finalize_transaction(self):
        """
        Handles the finalization of the transaction, including payment processing.
        This method is intended to be overridden with specific payment processing logic.
        """
        pass
