import pandas as pd
import pytest

from ml.src import config


@pytest.fixture(scope="session")
def base() -> pd.DataFrame:
    if not config.ANALYTICAL_BASE_FILE.exists():
        pytest.skip("base analítica não gerada — rode build_analytical_base")
    return pd.read_parquet(config.ANALYTICAL_BASE_FILE)


@pytest.fixture(scope="session")
def sample(base) -> pd.DataFrame:
    return base.sample(400, random_state=0).reset_index(drop=True)
