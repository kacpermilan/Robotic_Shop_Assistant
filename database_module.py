import pyodbc


class DatabaseModule:
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
        self.__load_known_products()
        self.__load_known_barcodes()

    def __load_known_barcodes(self):
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
