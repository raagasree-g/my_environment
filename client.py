from openenv import OpenEnvClient

def get_client():
    return OpenEnvClient(base_url="http://localhost:8000")