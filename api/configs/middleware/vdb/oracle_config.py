from typing import Literal, Self

from pydantic import Field, NonNegativeInt, PositiveInt, model_validator
from pydantic_settings import BaseSettings


class OracleConfig(BaseSettings):
    """
    Configuration settings for the Oracle database.
    """

    ORACLE_USER: str | None = Field(
        description="Username for authenticating with the Oracle database",
        default=None,
    )

    ORACLE_PASSWORD: str | None = Field(
        description="Password for authenticating with the Oracle database",
        default=None,
    )

    ORACLE_DSN: str | None = Field(
        description="Oracle database connection string. For a traditional database, use format "
        "'host:port/service_name'. For an autonomous database, use the service name from tnsnames.ora in the wallet",
        default=None,
    )

    ORACLE_CONFIG_DIR: str | None = Field(
        description="Directory containing the tnsnames.ora configuration file. Only used in thin mode connection",
        default=None,
    )

    ORACLE_WALLET_LOCATION: str | None = Field(
        description="Oracle wallet directory path containing the wallet files for secure connection",
        default=None,
    )

    ORACLE_WALLET_PASSWORD: str | None = Field(
        description="Password to decrypt the Oracle wallet, if it is encrypted",
        default=None,
    )

    ORACLE_IS_AUTONOMOUS: bool = Field(
        description="Flag indicating whether connecting to Oracle Autonomous Database",
        default=False,
    )

    ORACLE_POOL_MIN: PositiveInt = Field(
        description="Minimum number of Oracle connections kept open in the pool",
        default=1,
    )

    ORACLE_POOL_MAX: PositiveInt = Field(
        description="Maximum number of Oracle connections allowed in the pool",
        default=5,
    )

    ORACLE_POOL_INCREMENT: PositiveInt = Field(
        description="Number of Oracle connections to add when the pool needs to grow",
        default=1,
    )

    ORACLE_POOL_PING_INTERVAL: NonNegativeInt = Field(
        description="Seconds before a pooled Oracle connection is pinged on acquire; 0 validates every checkout",
        default=0,
    )

    ORACLE_ENABLE_VECTOR_INDEX: bool = Field(
        description="Create an approximate Oracle VECTOR index after initial Knowledge embeddings are stored",
        default=False,
    )

    ORACLE_VECTOR_INDEX_TYPE: Literal["HNSW", "IVF"] = Field(
        description="Oracle VECTOR index type; HNSW requires a configured Vector Pool on self-managed databases",
        default="IVF",
    )

    ORACLE_VECTOR_INDEX_DISTANCE: Literal["COSINE"] = Field(
        description="Distance metric for Oracle Knowledge vector indexes; must match COSINE retrieval semantics",
        default="COSINE",
    )

    ORACLE_VECTOR_INDEX_ACCURACY: int = Field(
        description="Target accuracy percentage for approximate Oracle vector search (1-100)",
        default=95,
    )

    ORACLE_VECTOR_INDEX_NEIGHBORS: int = Field(
        description="HNSW maximum neighbors per vector (2-2048)",
        default=32,
    )

    ORACLE_VECTOR_INDEX_EFCONSTRUCTION: int = Field(
        description="HNSW construction candidate limit (1-65535)",
        default=200,
    )

    ORACLE_VECTOR_INDEX_PARTITIONS: int = Field(
        description="IVF target number of neighbor partitions (1-10000000)",
        default=64,
    )

    @model_validator(mode="after")
    def validate_vector_index_config(self) -> Self:
        if not self.ORACLE_ENABLE_VECTOR_INDEX:
            return self
        if not 1 <= self.ORACLE_VECTOR_INDEX_ACCURACY <= 100:
            raise ValueError("ORACLE_VECTOR_INDEX_ACCURACY must be between 1 and 100")
        if self.ORACLE_VECTOR_INDEX_TYPE == "HNSW":
            if not 2 <= self.ORACLE_VECTOR_INDEX_NEIGHBORS <= 2048:
                raise ValueError("ORACLE_VECTOR_INDEX_NEIGHBORS must be between 2 and 2048 for HNSW")
            if not 1 <= self.ORACLE_VECTOR_INDEX_EFCONSTRUCTION <= 65535:
                raise ValueError("ORACLE_VECTOR_INDEX_EFCONSTRUCTION must be between 1 and 65535 for HNSW")
        elif not 1 <= self.ORACLE_VECTOR_INDEX_PARTITIONS <= 10_000_000:
            raise ValueError("ORACLE_VECTOR_INDEX_PARTITIONS must be between 1 and 10000000 for IVF")
        return self
