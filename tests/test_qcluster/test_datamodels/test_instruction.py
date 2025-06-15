import copy
import unittest

import numpy as np

from qcluster.custom_types import ClusterDescription
from qcluster.datamodels.instruction import Instruction, InstructionCollection
from qcluster.datamodels.sample import Sample, SampleCollection


class TestInstruction(unittest.TestCase):
    """Tests for the Instruction data class."""

    def test_init(self):
        """Test basic object creation."""
        instr = Instruction(id=1, text="test instruction")
        self.assertEqual(instr.id, 1)
        self.assertEqual(instr.text, "test instruction")
        self.assertIsNone(instr.embedding)
        self.assertIsNone(instr.cluster)

    def test_embedding_shape(self):
        """Test the embedding_shape property."""
        instr_no_embedding = Instruction(id=1, text="test")
        self.assertIsNone(instr_no_embedding.embedding_shape)

        embedding = np.array([1, 2, 3])
        instr_with_embedding = Instruction(id=2, text="test 2", embedding=embedding)
        self.assertEqual(instr_with_embedding.embedding_shape, (3,))

        # Test with a non-numpy object that has no 'shape' attribute
        instr_with_list_embedding = Instruction(
            id=3, text="test 3", embedding=[1, 2, 3]
        )
        self.assertIsNone(instr_with_list_embedding.embedding_shape)

    def test_from_sample(self):
        """Test creating an Instruction from a Sample object."""
        sample = Sample(
            id=10,
            instruction="instruction from sample",
            flags="test_flag",
            category="ACCOUNT",
            intent="test_intent",
            response="test response",
        )
        instr = Instruction.from_sample(sample)
        self.assertEqual(instr.id, 10)
        self.assertEqual(instr.text, "instruction from sample")
        self.assertIsNone(instr.embedding)
        self.assertIsNone(instr.cluster)

    def test_update_embedding(self):
        """Test updating an instruction's embedding."""
        instr = Instruction(id=1, text="test embedding")
        # A simple embedding function for testing
        def embedding_func(texts):
            return np.array([hash(t) for t in texts])
        instr.update_embedding(embedding_func)
        self.assertIsNotNone(instr.embedding)
        self.assertIsInstance(instr.embedding, np.ndarray)
        self.assertEqual(instr.embedding.shape, (1,))
        self.assertEqual(instr.embedding[0], hash("test embedding"))

    def test_repr(self):
        """Test the string representation of an Instruction."""
        instr1 = Instruction(id=1, text="test text")
        self.assertEqual(
            repr(instr1),
            "Instruction(id=1, text='test text', embedding_shape=NA, cluster=None)",
        )

        embedding = np.array([[1, 2], [3, 4]])
        instr2 = Instruction(id=2, text="test 2", embedding=embedding, cluster=5)
        self.assertEqual(
            repr(instr2),
            "Instruction(id=2, text='test 2', embedding_shape=(2, 2), cluster=5)",
        )


class TestInstructionCollection(unittest.TestCase):
    """Tests for the InstructionCollection class."""

    def setUp(self):
        """Set up a collection of instructions for testing."""
        self.instructions_data = [
            {"id": 10, "text": "instruction 1", "cluster": 0},
            {"id": 20, "text": "instruction 2", "cluster": 1},
            {"id": 30, "text": "instruction 3", "cluster": 0},
            {"id": 40, "text": "instruction 4", "cluster": 1},
            {"id": 50, "text": "instruction 5", "cluster": None},
        ]
        self.instructions = [Instruction(**data) for data in self.instructions_data]
        self.collection = InstructionCollection(instructions=self.instructions)

    def test_count(self):
        self.assertEqual(self.collection.count, 5)

    def test_clusters(self):
        """Test the `clusters` property."""
        self.assertEqual(self.collection.clusters, [0, 1, 0, 1, None])
        empty_collection = InstructionCollection(instructions=[])
        self.assertEqual(empty_collection.clusters, [])

    def test_embeddings(self):
        """Test the `embeddings` property."""
        for i, inst in enumerate(self.collection.instructions, start=1):
            inst.embedding = np.array([i])
        embeddings = self.collection.embeddings
        self.assertEqual(len(embeddings), len(self.collection.instructions))
        self.assertTrue(np.array_equal(embeddings[0], np.array([1])))
        self.assertTrue(np.array_equal(embeddings[2], np.array([3])))

    def test_from_samples(self):
        """Test creating a collection from a list of Samples."""
        samples = [
            Sample(
                id=1,
                instruction="sample 1",
                flags="test_flag",
                category="ACCOUNT",
                intent="test_intent",
                response="test response",
            ),
            Sample(
                id=2,
                instruction="sample 2",
                flags="test_flag_2",
                category="ORDER",
                intent="test_intent_2",
                response="test response 2",
            ),
        ]
        sample_collection = SampleCollection(samples=samples)
        collection = InstructionCollection.from_samples(sample_collection)
        self.assertEqual(len(collection), 2)
        self.assertEqual(collection[0].id, 1)
        self.assertEqual(collection[1].text, "sample 2")

    def test_update_embeddings(self):
        """Test updating embeddings for the entire collection."""
        def embedding_func(texts):
            return [np.array([len(t)]) for t in texts]
        self.collection.update_embeddings(embedding_func)
        self.assertIsNotNone(self.collection.instructions[0].embedding)
        self.assertEqual(
            self.collection.instructions[0].embedding[0], len("instruction 1")
        )
        self.assertEqual(
            self.collection.instructions[4].embedding[0], len("instruction 5")
        )

    def test_cluster(self):
        """Test applying a clustering function to the collection."""
        # Add embeddings to a subset of instructions
        self.collection.instructions[0].embedding = np.array([1.0])
        self.collection.instructions[2].embedding = np.array([2.0])
        self.collection.instructions[4].embedding = np.array([3.0])
        with self.assertRaises(ValueError):
            _ = self.collection.cluster

    def test_update_clusters(self):
        """Test the end-to-end process of updating embeddings and clusters."""
        def clustering_func(embeddings):
            return [e[0] % 2 for e in embeddings]
        coll = copy.deepcopy(self.collection[0:4])  # Only take the first 4 instructions
        with self.assertRaises(ValueError):
            self.collection.update_clusters(
                clustering_function=clustering_func, use_raw_instructions=False
            )
        coll.update_embeddings(
            embedding_function=lambda texts: [np.array([len(t)]) for t in texts]
        )
        coll.update_clusters(
            clustering_function=clustering_func, use_raw_instructions=False
        )
        for inst in coll.instructions:
            self.assertIsNotNone(inst.embedding)
            self.assertIsNotNone(inst.cluster)
            self.assertEqual(inst.cluster, len(inst.text) % 2)

    def test_len(self):
        self.assertEqual(len(self.collection), 5)
        self.assertEqual(len(InstructionCollection(instructions=[])), 0)

    def test_iter(self):
        """Test iteration over the collection."""
        inst_ids = [i.id for i in self.collection]
        self.assertEqual(inst_ids, [10, 20, 30, 40, 50])

    def test_getitem(self):
        """Test getting items by ID, index, and slice."""
        # By ID
        self.assertEqual(self.collection[0].text, "instruction 1")
        self.assertEqual([i.text for i in self.collection[-2:-1]], ["instruction 4"])
        with self.assertRaises(IndexError):
            _ = self.collection[99]  # Non-existent ID

    def test_to_list_of_strings(self):
        self.assertEqual(
            self.collection.to_list_of_strings(),
            [
                "instruction 1",
                "instruction 2",
                "instruction 3",
                "instruction 4",
                "instruction 5",
            ],
        )

    def test_get_cluster(self):
        """Test filtering the collection by cluster ID."""
        cluster_0_coll = self.collection.get_cluster(0)
        self.assertIsInstance(cluster_0_coll, InstructionCollection)
        self.assertEqual(len(cluster_0_coll), 2)
        self.assertTrue(all(i.cluster == 0 for i in cluster_0_coll))

        # Test getting a non-existent cluster
        cluster_99_coll = self.collection.get_cluster(99)
        self.assertEqual(len(cluster_99_coll), 0)

    def test_group_by_cluster(self):
        """Test grouping instructions by cluster ID."""
        collection = self.collection[0:4]  # Only take the first 4 instructions
        grouped = collection.group_by_cluster()
        self.assertEqual(sorted(grouped.keys()), [0, 1])
        self.assertEqual(len(grouped[0]), 2)
        self.assertEqual(len(grouped[1]), 2)
        self.assertEqual(grouped[0][0].id, 10)
        self.assertEqual(grouped[1][0].id, 20)

    def test_is_a_cluster(self):
        """Test the is_a_cluster property."""
        self.assertFalse(self.collection.is_a_cluster())

        cluster_0_coll = self.collection.get_cluster(0)
        self.assertTrue(cluster_0_coll.is_a_cluster)

        # Collection with a None cluster should not be considered a single cluster
        collection_with_none = InstructionCollection(
            instructions=[Instruction(id=1, text="t")]
        )
        self.assertFalse(collection_with_none.is_a_cluster())

        # Empty collection is not a cluster
        empty_collection = InstructionCollection(instructions=[])
        self.assertFalse(empty_collection.is_a_cluster())

    def test_describe(self):
        """Test generating a description for a cluster."""
        cluster_0_coll = self.collection.get_cluster(0)

        def mock_desc_func(text: str) -> ClusterDescription:
            return ClusterDescription(title=f"Title: {text[:5]}", description="Desc")

        cluster_0_coll.describe(mock_desc_func)

        self.assertEqual(cluster_0_coll.description, "Desc")
        # The input text to the function is "instruction 1 and instruction 3"
        self.assertEqual(cluster_0_coll.title, "Title: instr")

    def test_description_embedding(self):
        """Test creating an embedding for the collection's description."""
        collection = InstructionCollection(
            instructions=[], description="test description"
        )
        def embedding_func(texts):
            return np.array([hash(t) for t in texts])
        embedding = collection.description_embedding(embedding_func)
        self.assertIsNotNone(embedding)
        self.assertEqual(embedding, hash("test description"))

    def test_centroid(self):
        """Test calculating the centroid of embeddings."""
        collection = self.collection.get_cluster(0)  # ids 10 and 30
        collection.instructions[0].embedding = np.array([1.0, 2.0, 3.0])
        collection.instructions[1].embedding = np.array([3.0, 4.0, 5.0])
        self.assertTrue(np.allclose(collection.centroid, np.array([2.0, 3.0, 4.0])))

    def test_centroid_edge_cases(self):
        """Test centroid calculation with missing embeddings."""
        # No embeddings in the collection
        collection = self.collection.get_cluster(1)
        with self.assertRaises(ValueError):
            _ = collection.centroid

        # Some embeddings in the collection
        collection = self.collection.get_cluster(0)
        collection.instructions[0].embedding = np.array([1.0, 2.0, 3.0])
        # instruction[1] (id=30) has no embedding
        with self.assertRaises(ValueError):
            _ = collection.centroid

        # No instructions in the collection
        _ = InstructionCollection(instructions=[])


if __name__ == "__main__":
    unittest.main()
