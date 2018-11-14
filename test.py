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
        torch.to(inpt, device=device, non_blocking=True), mask),
        torch.to(outpt, device=device, non_blocking=True),
    )


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

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
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

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm.tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0)
    )

    @trainer.on(ignite.engine.Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_interval == 0:
            pbar.desc = desc.format(engine.state.output)
            pbar.update(log_interval)

    @trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        tqdm.tqdm.write(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
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
            f"Avg accuracy: {avg_accuracy:.2f} Avg loss: {avg_nll:.2f}"
        )

        pbar.n = pbar.last_print_n = 0

    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()


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
