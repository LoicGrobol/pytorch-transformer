import argparse

import ignite.engine
import ignite.metrics

import torch
import torch.optim

import torchtext.data
import torchtext.datasets
import torchtext.vocab

import tqdm

import transformer


class Net(torch.nn.Module):
    def __init__(self, out_dim, vocab_size=10000, embdeddings_dim=300, pretrained_embeddings=None):
        super().__init__()
        if pretrained_embeddings is None:
            self.embeddings = torch.nn.Embedding(
                vocab_size,
                self.embeddings_dim,
            )
        else:
            self.embeddings = torch.nn.Embedding.from_pretrained(
                pretrained_embeddings,
            )
        self.encoder = transformer.Encoder(300, 300)
        self.out = torch.nn.Linear(300, out_dim)

    def forward(self, inpt):
        x, mask = inpt
        x = self.embeddings(x)
        scores = self.out(self.encoder(x, mask)[:, 0, ...])
        return torch.nn.functional.log_softmax(scores, dim=-1)


def prepare_batch(batch, device, non_blocking):
    (inpt, length), outpt = batch
    seq_len = length.max()
    mask = torch.zeros(length.size(0), seq_len, seq_len, dtype=torch.uint8, device=device)
    for i, l in enumerate(length):
        mask[i, :, l:] = 1
    return (
        (inpt.to(device=device, non_blocking=True), mask),
        outpt.to(device=device, non_blocking=True),
    )


def add_epoch_bar(engine):
    @engine.on(ignite.engine.Events.EPOCH_STARTED)
    def epoch_init(engine):
        if engine.state.max_epochs > 1:
            desc = f'Epoch {engine.state.epoch}/{engine.state.max_epochs}'
        else:
            desc = 'Running model'

        engine.state.epoch_bar = tqdm.tqdm(
            desc=desc,
            initial=0,
            total=len(engine.state.dataloader),
            unit='batch',
            dynamic_ncols=True,
            leave=False,
            unit_scale=True,
            mininterval=1,
        )

    @engine.on(ignite.engine.Events.ITERATION_COMPLETED)
    def update_bars(engine):
        engine.state.epoch_bar.update()

    @engine.on(ignite.engine.Events.EPOCH_COMPLETED)
    def epoch_feedback(engine):
        engine.state.epoch_bar.close()


def add_train_bar(engine):
    @engine.on(ignite.engine.Events.STARTED)
    def train_init(engine):
        engine.state.train_bar = tqdm.tqdm(
            desc='Training',
            initial=0,
            total=engine.state.max_epochs,
            unit='epoch',
            dynamic_ncols=True,
            leave=False,
            unit_scale=True,
            mininterval=1,
        )

    @engine.on(ignite.engine.Events.ITERATION_COMPLETED)
    def update_bars(engine):
        engine.state.train_bar.update(1/len(engine.state.dataloader))

    @engine.on(ignite.engine.Events.COMPLETED)
    def epoch_feedback(engine):
        engine.state.train_bar.close()


def get_data_loaders(train_batch_size, test_batch_size, vectors, device):
    # set up fields
    TEXT = torchtext.data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = torchtext.data.Field(sequential=False, is_target=True)

    train_data, test_data = torchtext.datasets.IMDB.splits(TEXT, LABEL)

    TEXT.build_vocab(train_data, vectors=vectors)
    LABEL.build_vocab(train_data)

    train_iter, test_iter = torchtext.data.BucketIterator.splits(
        (train_data, test_data),
        batch_sizes=(train_batch_size, test_batch_size),
    )

    return train_iter, test_iter


def run(train_batch_size, val_batch_size, epochs, lr, momentum, log_interval, device):
    vectors = torchtext.vocab.GloVe(name='6B', dim=300)
    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size, vectors, device)
    model = Net(out_dim=3, pretrained_embeddings=vectors.vectors).to(device)

    optimizer = transformer.TransformerOpt(model.parameters(), 300)
    trainer = ignite.engine.create_supervised_trainer(
        model,
        optimizer,
        torch.nn.functional.nll_loss,
        prepare_batch=prepare_batch,
        device=device,
    )
    evaluator = ignite.engine.create_supervised_evaluator(
        model,
        metrics={
            'accuracy': ignite.metrics.Accuracy(),
            'nll': ignite.metrics.Loss(torch.nn.functional.nll_loss),
        },
        prepare_batch=prepare_batch,
        device=device,
    )
    add_train_bar(trainer)
    add_epoch_bar(trainer)
    add_epoch_bar(evaluator)


    @trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        tqdm.tqdm.write(
            "Training Results - Epoch: {}  Avg accuracy: {:.6f} Avg loss: {:.6f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll)
        )

    @trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        tqdm.tqdm.write(
            f"Validation Results - Epoch: {engine.state.epoch}  "
            f"Avg accuracy: {avg_accuracy:.6f} Avg loss: {avg_nll:.6f}"
        )

    trainer.run(train_loader, max_epochs=epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--val_batch_size', type=int, default=10,
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()

run(args.batch_size, args.val_batch_size, args.epochs, args.lr, args.momentum, args.log_interval,
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
