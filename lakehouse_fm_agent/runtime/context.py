
class Context:
    def __init__(self, spark=None):
        self.spark = spark
        self.current_df = None
