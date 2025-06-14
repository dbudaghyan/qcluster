from pydantic import BaseModel


class ClusterOutput(BaseModel):
    """
    Represents the cluster output for a sample instruction.
    """

    cluster_id: int
    name: str
    description: str
    count: int

    def __repr__(self) -> str:
        """
        Get a string representation of the cluster output.

        Returns:
            str: A formatted string representing the cluster output.
        """
        return (
            f"{self.__class__.__name__}(id={self.cluster_id}, "
            f"name='{self.name}', description='{self.description}', "
            f"count={self.count})"
        )

    def __str__(self) -> str:
        return self.__repr__()


class ClusterOutputCollection(BaseModel):
    """
    Represents a collection of cluster outputs.
    """

    outputs: list[ClusterOutput]

    def __len__(self) -> int:
        """
        Get the number of cluster outputs.

        Returns:
            int: The number of cluster outputs.
        """
        return len(self.outputs)

    def __repr__(self) -> str:
        """
        Get a string representation of the cluster output collection.

        Returns:
            str: A formatted string representing the cluster output collection.
        """
        indented_outputs = "\n".join(f"    {output!r}" for output in self.outputs)
        return (
            f"{self.__class__.__name__}(\n"
            f"  num_outputs={len(self)},\n"
            f"  outputs=[\n"
            f"{indented_outputs[:20]}\n....."
            f"  ]\n"
            f")"
        )

    def __str__(self) -> str:
        return self.__repr__()
