import os

from openenv.core.env_server.http_server import create_app

from models import CustomerSupportAction, CustomerSupportObservation
from server.environment import CustomerSupportOpenEnv


app = create_app(
    CustomerSupportOpenEnv,
    CustomerSupportAction,
    CustomerSupportObservation,
    env_name="intelligent_customer_support_decision_system",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int | None = None):
    import uvicorn

    resolved_port = int(port or os.getenv("PORT", "7860"))
    uvicorn.run(app, host=host, port=resolved_port)


if __name__ == "__main__":
    main()
