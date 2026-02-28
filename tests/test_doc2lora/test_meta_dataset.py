from continual_learning.doc2lora.meta_dataset import SyntheticMetaDataset


class TestSyntheticMetaDataset:
    def setup_method(self):
        self.dataset = SyntheticMetaDataset(num_items=50, seed=42)

    def test_length(self):
        assert len(self.dataset) == 50

    def test_item_has_required_keys(self):
        item = self.dataset[0]
        assert "context" in item
        assert "question" in item
        assert "answer" in item

    def test_context_is_nonempty(self):
        for i in range(min(10, len(self.dataset))):
            assert len(self.dataset[i]["context"]) > 0

    def test_question_is_nonempty(self):
        for i in range(min(10, len(self.dataset))):
            assert len(self.dataset[i]["question"]) > 0

    def test_answer_is_nonempty(self):
        for i in range(min(10, len(self.dataset))):
            assert len(self.dataset[i]["answer"]) > 0

    def test_deterministic_with_same_seed(self):
        ds1 = SyntheticMetaDataset(num_items=10, seed=42)
        ds2 = SyntheticMetaDataset(num_items=10, seed=42)
        assert ds1[0] == ds2[0]

    def test_different_with_different_seed(self):
        ds1 = SyntheticMetaDataset(num_items=10, seed=42)
        ds2 = SyntheticMetaDataset(num_items=10, seed=99)
        assert ds1[0] != ds2[0]

    def test_answer_appears_in_context(self):
        """Answers should be extractable from context."""
        item = self.dataset[0]
        assert item["answer"] in item["context"]
