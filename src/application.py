# server configuration!

PORT = 5000

HOST = "0.0.0.0"

# For Deployment
class DataSource:
    def __init__(self):
        self.host = "db-pf7p3.vpc-cdb.ntruss.com"
        self.port = 3306
        self.database="damdda"
        self.username="damdda"
        self.password="damdda123!@#"
        
# For Development
# class DataSource:
#     def __init__(self):
#         self.host = "localhost"
#         self.port = 6506
#         self.database="damdda"
#         self.username="damdda"
#         self.password="damdda"