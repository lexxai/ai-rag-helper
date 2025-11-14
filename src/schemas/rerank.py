from pydantic import BaseModel, Field
from config.settings import settings

example_model_name = settings.default_model_names["rerank"]


class RerankRequest(BaseModel):
    query: str = Field(examples=["How many people live in Berlin?"])
    candidates: list[str] = Field(
        examples=[
            [
                "hello world",
                "Kyiv is the capital city of Ukraine.",
                "London is the capital city of England and the United Kingdom with an estimated population of 8.9 million.",
                "this is a test",
                "Berlin is well known for its museums.",
                "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
                "this is another sentence",
            ]
        ]
    )
    model: str | None = Field(default=None, examples=[example_model_name])
    batch_size: int | None = Field(default=None, description="Batch size to use for inference")


class RerankResponse(BaseModel):
    scores: list[float] = Field(
        examples=[
            [
                -10.462787628173828,
                -11.19316291809082,
                -4.849969863891602,
                -10.884675979614258,
                -4.320077896118164,
                8.607139587402344,
                -10.74217414855957,
            ]
        ]
    )
    sorted_candidates: list[str] = Field(
        examples=[
            [
                "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
                "Berlin is well known for its museums.",
                "London is the capital city of England and the United Kingdom with an estimated population of 8.9 million.",
                "hello world",
                "this is another sentence",
                "this is a test",
                "Kyiv is the capital city of Ukraine.",
            ],
        ]
    )
    model_name: str = Field(examples=[example_model_name])
