from utils.rl_utils import PGTrainer
from utils.model_utils import get_router_model


if __name__ == '__main__':
    model = get_router_model(2, 'bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    trainer = PGTrainer(model, tokenizer, train_dataloader, test_dataloader, batch_size)
    trainer.train(num_iters=1)
