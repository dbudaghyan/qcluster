from qcluster.datamodels.output import ClusterOutput, ClusterOutputCollection


# --- Tests for ClusterOutput Class ---

def test_cluster_output_creation():
    """Tests the creation of a ClusterOutput object."""
    output = ClusterOutput(
        cluster_id=1,
        name="Test Name",
        description="Test Description",
        count=10,
    )
    assert output.cluster_id == 1
    assert output.name == "Test Name"
    assert output.description == "Test Description"
    assert output.count == 10


def test_cluster_output_creation_edge_cases():
    """Tests the creation of a ClusterOutput object with edge case values."""
    # Test with zero `count`
    output_zero_count = ClusterOutput(
        cluster_id=0,
        name="Zero Count",
        description="A cluster with zero items.",
        count=0,
    )
    assert output_zero_count.cluster_id == 0
    assert output_zero_count.name == "Zero Count"
    assert output_zero_count.description == "A cluster with zero items."
    assert output_zero_count.count == 0

    # Test with empty strings for name and description
    output_empty_strings = ClusterOutput(
        cluster_id=2,
        name="",
        description="",
        count=5,
    )
    assert output_empty_strings.name == ""
    assert output_empty_strings.description == ""


def test_cluster_output_repr_and_str():
    """Tests the string representation of a ClusterOutput."""
    output = ClusterOutput(
        cluster_id=1,
        name="Test Name",
        description="A cool description.",
        count=10,
    )
    expected_repr = ("ClusterOutput(id=1, name='Test Name',"
                     " description='A cool description.', count=10)")
    assert repr(output) == expected_repr
    assert str(output) == expected_repr


# --- Tests for ClusterOutputCollection Class ---

def create_cluster_output_list(num_outputs=3):
    """Helper function to create a list of ClusterOutput objects for testing."""
    return [
        ClusterOutput(
            cluster_id=i,
            name=f"Name {i}",
            description=f"Description {i}",
            count=10 * i,
        )
        for i in range(num_outputs)
    ]


def test_cluster_output_collection_creation():
    """Tests the creation of a ClusterOutputCollection object."""
    outputs = create_cluster_output_list(3)
    collection = ClusterOutputCollection(outputs=outputs)
    assert len(collection.outputs) == 3
    assert collection.outputs == outputs


def test_cluster_output_collection_creation_empty():
    """Tests the creation of an empty ClusterOutputCollection object."""
    collection = ClusterOutputCollection(outputs=[])
    assert len(collection.outputs) == 0
    assert len(collection) == 0


def test_cluster_output_collection_len():
    """Tests the __len__ method of ClusterOutputCollection."""
    # Test with multiple outputs
    outputs = create_cluster_output_list(5)
    collection = ClusterOutputCollection(outputs=outputs)
    assert len(collection) == 5

    # Test with one output
    outputs_single = create_cluster_output_list(1)
    collection_single = ClusterOutputCollection(outputs=outputs_single)
    assert len(collection_single) == 1

    # Test with no outputs
    collection_empty = ClusterOutputCollection(outputs=[])
    assert len(collection_empty) == 0


def test_cluster_output_collection_repr_and_str():
    """Tests the string representation of a ClusterOutputCollection."""
    outputs = create_cluster_output_list(2)
    collection = ClusterOutputCollection(outputs=outputs)
    repr_str = repr(collection)

    assert "ClusterOutputCollection(" in repr_str
    assert "num_outputs=2" in repr_str
    assert "outputs=[" in repr_str
    assert "....." in repr_str
    assert str(collection) == repr_str


def test_cluster_output_collection_repr_and_str_empty():
    """Tests the string representation of an empty ClusterOutputCollection."""
    collection = ClusterOutputCollection(outputs=[])
    repr_str = repr(collection)
    assert "ClusterOutputCollection(" in repr_str
    assert "num_outputs=0" in repr_str
    assert "outputs=[\n\n....." in repr_str
    assert str(collection) == repr_str
