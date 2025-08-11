from utils.rl_utils import PGTrainer
from utils.model_utils import get_router_model
from transformers import AutoTokenizer


def get_router_dataset():
    return ['I am training ...'] * 1000, ['I am testing ...'] * 100


if __name__ == '__main__':
    model = get_router_model(2, 'google-bert/bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')

    train_dataset, test_dataset = get_router_dataset()
    trainer = PGTrainer(model, tokenizer, train_dataset, test_dataset, batch_size=4)
    trainer.train(num_iters=1)
