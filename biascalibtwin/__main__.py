import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn
from torch.nn import functional as F


class Twin(nn.Module):
    def __init__(self, c, sensor_count, control_count, observation, a, b):
        super().__init__()
        self.c = c
        h = c['width']
        self.sim = nn.GRUCell(8 + control_count, h)
        self.online = nn.GRUCell(3 * sensor_count + control_count, h)
        self.bias_increment = nn.Sequential(nn.Linear(h, 64), nn.Tanh(), nn.Linear(64, 4))
        self.sim_precision = nn.Linear(h, 4)
        self.online_message = nn.Linear(h, 8)
        self.metro_precision = nn.Parameter(torch.zeros(2))
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(h + 5, 32), nn.ReLU(), nn.Linear(32, 1)) for _ in range(3)])
        self.transition = nn.Sequential(nn.Linear(4 + control_count, h), nn.Tanh(), nn.Linear(h, 4))
        self.process_noise = nn.Parameter(torch.full((4,), -3.))
        self.register_buffer('observation', torch.tensor(observation, dtype=torch.float32))
        self.register_buffer('a', torch.tensor(a, dtype=torch.float32))
        self.register_buffer('b', torch.tensor(b, dtype=torch.float32))

    def forward(self, data):
        sim, sensor, mask, age, control, metro, metro_mask = [data[k].float() for k in ['sim', 'sensor', 'mask', 'age', 'control', 'metro', 'metro_mask']]
        count, length, _ = sim.shape
        mean = sim.new_zeros(count, 4)
        variance = sim.new_ones(count, 4)
        bias = sim.new_zeros(count, 4)
        hs = sim.new_zeros(count, self.c['width'])
        ho = hs.clone()
        means, variances, reliabilities, shares, message_means, priors = [], [], [], [], [], []
        for t in range(length):
            prior = mean @ self.a.T + control[:, t] @ self.b.T + self.transition(torch.cat([mean, control[:, t]], 1))
            prior_variance = variance @ self.a.square().T + F.softplus(self.process_noise) + 1e-6
            hs = self.sim(torch.cat([sim[:, t], control[:, t], mean], 1), hs)
            bias = bias + self.c['bias_step'] * self.bias_increment(hs)
            if self.c['variant'] == 'no-bias':
                bias = bias * 0
            ho = self.online(torch.cat([sensor[:, t] * mask[:, t], mask[:, t], age[:, t], control[:, t]], 1), ho)
            online = self.online_message(ho)
            metro_mean = torch.cat([sim.new_zeros(count, 2), metro[:, t]], 1)
            metro_p = torch.cat([sim.new_zeros(count, 2), (F.softplus(self.metro_precision) + 1e-6) * metro_mask[:, t]], 1)
            source_mean = torch.stack([sim[:, t] - bias, online[:, :4], metro_mean], 1)
            precision = torch.stack([F.softplus(self.sim_precision(hs)) + 1e-6,
                                     (F.softplus(online[:, 4:]) + 1e-6) * mask[:, t].mean(1, keepdim=True), metro_p], 1)
            if self.c['variant'] == 'sensor-only':
                precision = precision * sim.new_tensor([0, 1, 0])[None, :, None]
            elif self.c['variant'] == 'simulator-only':
                precision = precision * sim.new_tensor([1, 0, 0])[None, :, None]
            gates = []
            for j in range(3):
                missing = 1 - (precision[:, j] > 0).float().mean(1)
                source_age = age[:, t].mean(1) if j == 1 else sim.new_zeros(count)
                innovation = ((source_mean[:, j] - prior).square() / prior_variance).mean(1)
                conflict = ((source_mean[:, j:j + 1] - source_mean).square() * (precision > 0)).sum((1, 2)) / (precision > 0).sum((1, 2)).clamp_min(1)
                physics = (source_mean[:, j] - prior).square().mean(1)
                quality = torch.stack([missing, source_age, innovation, conflict, physics], 1)
                state = ho if j == 1 else hs
                reliability = .02 + .98 * self.gates[j](torch.cat([quality, state], 1)).sigmoid().squeeze(1)
                gates.append(reliability)
            reliability = torch.stack(gates, 1)
            if self.c['variant'] == 'fixed-gates':
                reliability = torch.ones_like(reliability)
            effective = reliability[..., None] * precision
            posterior_precision = 1 / prior_variance + effective.sum(1)
            variance = posterior_precision.reciprocal()
            mean = variance * (prior / prior_variance + (effective * source_mean).sum(1))
            means.append(mean)
            variances.append(variance)
            reliabilities.append(reliability)
            shares.append(effective.sum(-1) / posterior_precision.sum(-1, keepdim=True))
            message_means.append(source_mean)
            priors.append(prior)
        return {k: torch.stack(v, 1) for k, v in dict(mean=means, variance=variances, reliability=reliabilities,
                                                     share=shares, messages=message_means, prior=priors).items()}

    def loss(self, data):
        result = self(data)
        mean, variance = result['mean'], result['variance']
        prediction = mean @ self.observation.T
        sensor_variance = variance @ self.observation.square().T + self.c['sensor_noise'] ** 2
        sensor_nll = .5 * ((data['sensor'] - prediction).square() / sensor_variance + sensor_variance.log())
        mask = data['mask']
        loss = (sensor_nll * mask).sum() / mask.sum().clamp_min(1)
        metro_mask = data['metro_mask']
        sigma = variance[:, :, 2:].sqrt().clamp_min(1e-6)
        residual = data['metro'] - mean[:, :, 2:]
        metro_nll = .5 * (residual.square() / sigma.square() + 2 * sigma.log())
        loss = loss + (metro_nll * metro_mask).sum() / metro_mask.sum().clamp_min(1)
        prior_variance = F.softplus(self.process_noise) + 1e-6
        kl = .5 * ((variance + (mean - result['prior']).square()) / prior_variance - 1 + prior_variance.log() - variance.log()).mean()
        loss = loss + self.c['kl_weight'] * kl
        if self.c['variant'] != 'no-physics':
            physics = mean[:, 1:] - mean[:, :-1] @ self.a.T - data['control'][:, 1:] @ self.b.T
            loss = loss + self.c['physics_weight'] * physics.square().mean()
        z = residual / sigma
        phi = torch.exp(-.5 * z.square()) / math.sqrt(2 * math.pi)
        crps = sigma * (z * (2 * torch.special.ndtr(z) - 1) + 2 * phi - 1 / math.sqrt(math.pi))
        loss = loss + self.c['calibration_weight'] * (crps * metro_mask).sum() / metro_mask.sum().clamp_min(1)
        observed = metro_mask.bool().all(-1)
        if observed.any():
            error = (result['messages'][:, :, :, 2:] - data['metro'][:, :, None]).square().sum(-1)
            target = torch.exp(-.5 * error / self.c['metro_noise'] ** 2).detach()
            gate_loss = F.binary_cross_entropy(result['reliability'][observed], target[observed])
            loss = loss + self.c['gate_weight'] * gate_loss
        return loss


def prepare(c, out, source):
    raw = dict(np.load(source, allow_pickle=False))
    required = {'sim', 'sensor', 'mask', 'age', 'control', 'metro', 'metro_mask', 'truth', 'run', 'split', 'observation'}
    if not required <= set(raw):
        raise ValueError('Missing arrays: ' + ','.join(sorted(required - set(raw))))
    if len(np.unique(raw['run'])) != len(raw['run']):
        raise ValueError('Provide each complete trajectory once')
    training = raw['split'] == 'train'
    state_mean = raw['sim'][training].mean((0, 1))
    state_scale = raw['sim'][training].std((0, 1)).clip(1e-6)
    for key in ['sim', 'truth']:
        raw[key] = (raw[key] - state_mean) / state_scale
    raw['metro'] = (raw['metro'] - state_mean[2:]) / state_scale[2:]
    sensor_offset = state_mean @ raw['observation'].T
    raw['sensor'] = raw['sensor'] - sensor_offset
    raw['observation'] = raw['observation'] * state_scale[None]
    previous = raw['sim'][training, :-1].reshape(-1, 4)
    u = raw['control'][training, 1:].reshape(-1, raw['control'].shape[-1])
    target = raw['sim'][training, 1:].reshape(-1, 4)
    coefficients = np.linalg.lstsq(np.concatenate([previous, u], 1), target, rcond=None)[0]
    raw['transition_a'], raw['transition_b'] = coefficients[:4].T, coefficients[4:].T
    raw['state_scale'], raw['state_mean'] = state_scale, state_mean
    raw['sensor'] = np.nan_to_num(raw['sensor'])
    raw['metro'] = np.nan_to_num(raw['metro'])
    np.savez_compressed(out / 'data.npz', **raw)


def build(c, d):
    return Twin(c, d['sensor'].shape[-1], d['control'].shape[-1], d['observation'], d['transition_a'], d['transition_b'])


def batch(d, ids):
    return {k: torch.tensor(d[k][ids], dtype=torch.float32) for k in ['sim', 'sensor', 'mask', 'age', 'control', 'metro', 'metro_mask']}


def train(c, out):
    torch.manual_seed(c['seed'])
    d = np.load(out / 'data.npz')
    model = build(c, d)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'])
    ids = np.flatnonzero(d['split'] == 'train')
    validation = np.flatnonzero(d['split'] == 'val')
    best, stale, curve = np.inf, 0, []
    for epoch in range(c['epochs']):
        model.train()
        for group in torch.tensor(ids)[torch.randperm(len(ids))].split(c['batch_size']):
            loss = model.loss(batch(d, group))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            value = float(model.loss(batch(d, validation)))
        curve.append(dict(epoch=epoch, validation_objective=value))
        if value < best:
            best, stale = value, 0
            torch.save(model.state_dict(), out / 'model.pt')
        else:
            stale += 1
        if stale >= c['patience']:
            break
    observed = d['metro'][ids][d['metro_mask'][ids].all(-1)]
    limits = np.quantile(observed, [.005, .995], axis=0)
    np.save(out / 'limits.npy', limits)
    pd.DataFrame(curve).to_csv(out / 'learning.csv', index=False)


def run(c, out):
    d = np.load(out / 'data.npz')
    model = build(c, d)
    model.load_state_dict(torch.load(out / 'model.pt', weights_only=True, map_location='cpu'))
    model.eval()
    saved = {}
    start = time.perf_counter()
    with torch.no_grad():
        for role in ['val', 'test']:
            ids = np.flatnonzero(d['split'] == role)
            result = model(batch(d, ids))
            for key, value in result.items():
                saved[role + '_' + key] = value.numpy()
            saved[role + '_ids'] = ids
    saved['elapsed_seconds'] = np.array(time.perf_counter() - start)
    np.savez_compressed(out / 'predictions.npz', **saved)


def evaluate(c, out):
    d, p = np.load(out / 'data.npz'), np.load(out / 'predictions.npz')
    ids = p['test_ids']
    mean, variance = p['test_mean'], p['test_variance']
    residual = d['truth'][ids] - mean
    unmeasured = ~d['metro_mask'][ids].astype(bool)
    sigma = np.sqrt(variance.clip(1e-12))
    z = residual / sigma
    crps = sigma * (z * (2 * ndtr(z) - 1) + 2 * np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi))
    comp = residual[:, :, 2:][unmeasured]
    coverage = (abs(residual) <= 1.96 * sigma).mean()
    result = dict(method=c['variant'], normalized_rmse=float(np.sqrt((comp ** 2).mean())),
                  normalized_mae=float(abs(comp).mean()), nll=float((.5 * (np.log(2 * np.pi * variance) + z ** 2)).mean()),
                  crps=float(crps.mean()), coverage95=float(coverage), coverage_error=float(abs(coverage - .95)),
                  physics_residual=float(((mean[:, 1:] - mean[:, :-1] @ d['transition_a'].T - d['control'][ids, 1:] @ d['transition_b'].T) ** 2).mean()))
    for j, name in enumerate(['simulator', 'sensor', 'metrology']):
        result[name + '_information_share'] = float(p['test_share'][:, :, j].mean())
        anchor = d['metro_mask'][ids].all(-1)
        error = ((p['test_messages'][:, :, j, 2:] - d['metro'][ids]) ** 2).sum(-1)
        result[name + '_reliability_correlation'] = float(spearmanr(p['test_reliability'][:, :, j][anchor], -error[anchor]).statistic)
    low, high = np.load(out / 'limits.npy')
    probability = 1 - np.prod(ndtr((high - mean[:, :, 2:]) / sigma[:, :, 2:]) - ndtr((low - mean[:, :, 2:]) / sigma[:, :, 2:]), axis=-1)
    validation_sigma = np.sqrt(p['val_variance'][:, :, 2:]).max(-1)
    abstention = float(np.quantile(validation_sigma, c['uncertainty_quantile']))
    accepted = sigma[:, :, 2:].max(-1) <= abstention
    truth = ((d['truth'][ids, :, 2:] < low) | (d['truth'][ids, :, 2:] > high)).any(-1)
    predicted = probability >= .8
    result.update(balanced_accuracy=balanced_accuracy_score(truth.ravel(), (predicted & accepted).ravel()),
                  f1=f1_score(truth.ravel(), (predicted & accepted).ravel(), zero_division=0),
                  false_alarm=float((predicted & accepted)[~truth].mean()), missed_detection=float((~predicted | ~accepted)[truth].mean()),
                  abstention_rate=float((~accepted).mean()))
    delays = []
    for target, alarm in zip(truth, predicted & accepted):
        starts = np.flatnonzero(target & ~np.r_[False, target[:-1]])
        for onset in starts:
            end = onset
            while end < len(target) and target[end]:
                end += 1
            hits = np.flatnonzero(alarm[onset:end])
            delays.append((hits[0] if len(hits) else end - onset) * c['sample_minutes'])
    result['detection_delay_minutes'] = float(np.mean(delays)) if delays else np.nan
    pd.DataFrame([result]).to_csv(out / 'metrics.csv', index=False)


def plot(out):
    p = np.load(out / 'predictions.npz')
    mean, sigma = p['test_mean'][0, :, 2], np.sqrt(p['test_variance'][0, :, 2])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(mean)
    ax.fill_between(np.arange(len(mean)), mean - 1.96 * sigma, mean + 1.96 * sigma, alpha=.2)
    ax.set(xlabel='Causal sample', ylabel='Normalized product-G composition')
    fig.tight_layout()
    fig.savefig(out / 'posterior.png', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'train', 'run', 'evaluate', 'plot'])
    p.add_argument('--input')
    p.add_argument('--config', default='config.json')
    p.add_argument('--output', default='runs/default')
    p.add_argument('--variant')
    args = p.parse_args()
    c = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.variant:
        c['variant'] = args.variant
    if c['variant'] not in ['full', 'no-bias', 'fixed-gates', 'no-physics', 'sensor-only', 'simulator-only']:
        p.error('Unknown variant')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    settings = out / 'settings.json'
    if args.variant is None and settings.exists():
        c['variant'] = json.loads(settings.read_text(encoding='utf-8')).get('variant', c.get('variant', 'full'))
    settings.write_text(json.dumps(c, indent=2))
    if args.command == 'prepare':
        if not args.input:
            p.error('--input required')
        prepare(c, out, args.input)
    elif args.command == 'plot':
        plot(out)
    else:
        globals()[args.command](c, out)


if __name__ == '__main__':
    main()
