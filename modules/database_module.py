import pyodbc


class DatabaseModule:
    """
    A class to handle database operations for managing product and barcode data.

    Attributes:
    -----------
    __conn_str : str
    The connection string for the database.
    __products_table : str
    The name of the table containing product data.
    __barcodes_table : str
    The name of the table containing barcode data.
    products : dict
    A dictionary to store product information.
    barcodes : dict
    A dictionary to store barcode information.

    Methods:
    --------
    refresh_data(self)
    Loads product and barcode data from the database into memory.

    __load_known_barcodes(self)
    Internal method to load barcode data from the database.

    __load_known_products(self)
    Internal method to load product data from the database.
    """
    def __init__(self, server, database, username, password, driver='{ODBC Driver 17 for SQL Server}'):
        self.__conn_str = (
            f'DRIVER={driver};'
            f'SERVER={server};'
            f'DATABASE={database};'
            f'UID={username};'
            f'PWD={password}'
        )

        self.__products_table = "products"
        self.__barcodes_table = "barcodes"
        self.products = {}
        self.barcodes = {}

    def refresh_data(self):
        """
        Loads the latest product and barcode data from the database.
        """
        self.__load_known_products()
        self.__load_known_barcodes()

    def __load_known_barcodes(self):
        """
        Internal method to load barcode data from the database.
        """
        self.barcodes = {}
        try:
            with pyodbc.connect(self.__conn_str) as conn:
                cursor = conn.cursor()
                cursor.execute(f'SELECT [Barcode], [ProductID] FROM {self.__barcodes_table}')

                for row in cursor:
                    barcode_data, associated_info = row
                    self.barcodes[barcode_data] = associated_info

        except pyodbc.Error as e:
            print(f"Database error: {e}")
        except Exception as e:
            print(f"Error loading known barcodes: {e}")

    def __load_known_products(self):
        """
        Internal method to load product data from the database.
        """
        self.products = {}
        try:
            with pyodbc.connect(self.__conn_str) as conn:
                cursor = conn.cursor()
                cursor.execute(f'SELECT [ID], [Name], [Price] FROM {self.__products_table}')

                for row in cursor:
                    product_id, product_name, product_price = row
                    self.products[product_id] = {
                        'name': product_name,
                        'price': product_price
                    }

        except pyodbc.Error as e:
            print(f"Database error: {e}")
        except Exception as e:
            print(f"Error loading known products: {e}")
