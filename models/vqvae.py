import math, pickle, os
import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from utils.dsp import *
import sys
import time
from layers.overtone import Overtone
from layers.vector_quant import *
from layers.downsampling_encoder import DownsamplingEncoder
import utils.env as env
import utils.logger as logger
import random

# from layers.singular_loss import SingularLoss

__model_factory = {
    'vqvae': VectorQuant,
    'vqvae_group': VectorQuantGroup,
}


def init_vq(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError("Unknown models: {}".format(name))
    return __model_factory[name](*args, **kwargs)


class Model(nn.Module):
    def __init__(self, model_type, rnn_dims, fc_dims, global_decoder_cond_dims, upsample_factors, num_group, num_sample,
                 normalize_vq=False, noise_x=False, noise_y=False):
        super().__init__()
        # self.n_classes = 256
        print("vqvae.py wavernn model definition params")
        print(
            f"wrnn_dims={rnn_dims}, fc_dims={fc_dims}, cond_channels={128}, global_cond_channels={global_decoder_cond_dims}")
        rnn_dims, fc_dims, 128, global_decoder_cond_dims
        self.overtone = Overtone(rnn_dims, fc_dims, 128, global_decoder_cond_dims)
        # self.vq = VectorQuant(1, 410, 128, normalize=normalize_vq)
        self.vq = init_vq(model_type, 1, num_group*num_sample, 128, num_group, num_sample, normalize=normalize_vq)
        self.noise_x = noise_x
        self.noise_y = noise_y
        encoder_layers = [
            (2, 4, 1),
            (2, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
        ]
        self.encoder = DownsamplingEncoder(128, encoder_layers)
        self.frame_advantage = 15
        self.num_params()

    def forward(self, global_decoder_cond, x, samples):
        # x: (N, 768, 3)
        # logger.log(f'x: {x.size()}')
        # samples: (N, 1022)
        # logger.log(f'samples: {samples.size()}')
        continuous = self.encoder(samples)
        # continuous: (N, 14, 64)
        # logger.log(f'continuous: {continuous.size()}')
        discrete, vq_pen, encoder_pen, entropy, _, _ = self.vq(continuous.unsqueeze(2))
        # discrete: (N, 14, 1, 64)
        # logger.log(f'discrete: {discrete.size()}')

        # cond: (N, 768, 64)
        # logger.log(f'cond: {cond.size()}')
        return self.overtone(x, discrete.squeeze(2), global_decoder_cond), vq_pen.mean(), encoder_pen.mean(), entropy

    def after_update(self):
        self.overtone.after_update()
        self.vq.after_update()

    def forward_generate(self, global_decoder_cond, samples, deterministic=False, use_half=False, verbose=False,
                         only_discrete=False):
        if use_half:
            samples = samples.half()
        # samples: (L)
        # logger.log(f'samples: {samples.size()}')
        self.eval()
        with torch.no_grad():
            continuous = self.encoder(samples)
            discrete, vq_pen, encoder_pen, entropy, index_atom, index_group = self.vq(continuous.unsqueeze(2))
            print("Inside forward_generate(), global_decoder_cond.size()", global_decoder_cond.size())  # [1, 30]
            print("Inside forward_generate(), discrete.size()", discrete.size())  # [1, 557, 1, 128]
            logger.log(f'entropy: {entropy}')
            # cond: (1, L1, 64)
            # logger.log(f'cond: {cond.size()}')
            if only_discrete:
                output = None
            else:
                output = self.overtone.generate(discrete.squeeze(2), global_decoder_cond, use_half=use_half,
                                                verbose=verbose)
        self.train()
        return output, discrete, index_atom, index_group

    def forward_generate_from_tokens(self, global_decoder_cond, discretes, verbose=False):
        print("Inside forward_generate_from_tokens(), global_decoder_cond.size()", global_decoder_cond.size())  # [1, 30]
        print("Inside forward_generate_from_tokens(), discretes.size()", discretes.size())  # [557, 128]
        discretes = discretes.unsqueeze(0)  # introduce N dimension [1, 557, 128]
        self.eval()
        with torch.no_grad():
            output = self.overtone.generate(discretes, global_decoder_cond, verbose=verbose)
        self.train()
        return output

    def num_params(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        logger.log('Trainable Parameters: %.3f million' % parameters)

    def load_state_dict(self, dict, strict=True):
        if strict:
            return super().load_state_dict(self.upgrade_state_dict(dict))
        else:
            my_dict = self.state_dict()
            new_dict = {}
            for key, val in dict.items():
                if key not in my_dict:
                    logger.log(f'Ignoring {key} because no such parameter exists')
                elif val.size() != my_dict[key].size():
                    logger.log(f'Ignoring {key} because of size mismatch')
                else:
                    logger.log(f'Loading {key}')
                    new_dict[key] = val
            return super().load_state_dict(new_dict, strict=False)

    def upgrade_state_dict(self, state_dict):
        out_dict = state_dict.copy()
        return out_dict

    def freeze_encoder(self):
        for name, param in self.named_parameters():
            if name.startswith('encoder.') or name.startswith('vq.'):
                logger.log(f'Freezing {name}')
                param.requires_grad = False
            else:
                logger.log(f'Not freezing {name}')

    def pad_left(self):
        return max(self.pad_left_decoder(), self.pad_left_encoder())

    def pad_left_decoder(self):
        return self.overtone.pad()

    def pad_left_encoder(self):
        return self.encoder.pad_left + (self.overtone.cond_pad - self.frame_advantage) * self.encoder.total_scale

    def pad_right(self):
        return self.frame_advantage * self.encoder.total_scale

    def total_scale(self):
        return self.encoder.total_scale

    def do_train(self, paths, dataset, optimiser, writer, epochs, test_epochs, batch_size, step, epoch, valid_index=[],
                 use_half=False, do_clip=False, beta=0.):

        if use_half:
            import apex
            optimiser = apex.fp16_utils.FP16_Optimizer(optimiser, dynamic_loss_scale=True)
        # for p in optimiser.param_groups : p['lr'] = lr
        criterion = nn.NLLLoss().cuda()
        # k = 0
        # saved_k = 0
        pad_left = self.pad_left()
        pad_left_encoder = self.pad_left_encoder()
        pad_left_decoder = self.pad_left_decoder()
        if self.noise_x:
            extra_pad_right = 127
        else:
            extra_pad_right = 0
        pad_right = self.pad_right() + extra_pad_right
        window = 16 * self.total_scale()
        logger.log(
            f'pad_left={pad_left_encoder}|{pad_left_decoder}, pad_right={pad_right}, total_scale={self.total_scale()}')

        for e in range(epoch, epochs):

            trn_loader = DataLoader(dataset, collate_fn=lambda batch: env.collate_multispeaker_samples(pad_left, window,
                                                                                                       pad_right,
                                                                                                       batch),
                                    batch_size=batch_size,
                                    num_workers=2, shuffle=True, pin_memory=True)

            start = time.time()
            running_loss_c = 0.
            running_loss_f = 0.
            running_loss_vq = 0.
            running_loss_vqc = 0.
            running_entropy = 0.
            running_max_grad = 0.
            running_max_grad_name = ""

            iters = len(trn_loader)

            for i, (speaker, wave16) in enumerate(trn_loader):

                speaker = speaker.cuda()
                wave16 = wave16.cuda()

                coarse = (wave16 + 2 ** 15) // 256
                fine = (wave16 + 2 ** 15) % 256

                coarse_f = coarse.float() / 127.5 - 1.
                fine_f = fine.float() / 127.5 - 1.
                total_f = (wave16.float() + 0.5) / 32767.5

                if self.noise_y:
                    noisy_f = total_f * (
                                0.02 * torch.randn(total_f.size(0), 1).cuda()).exp() + 0.003 * torch.randn_like(total_f)
                else:
                    noisy_f = total_f

                if use_half:
                    coarse_f = coarse_f.half()
                    fine_f = fine_f.half()
                    noisy_f = noisy_f.half()

                x = torch.cat([
                    coarse_f[:, pad_left - pad_left_decoder:-pad_right].unsqueeze(-1),
                    fine_f[:, pad_left - pad_left_decoder:-pad_right].unsqueeze(-1),
                    coarse_f[:, pad_left - pad_left_decoder + 1:1 - pad_right].unsqueeze(-1),
                ], dim=2)
                y_coarse = coarse[:, pad_left + 1:1 - pad_right]
                y_fine = fine[:, pad_left + 1:1 - pad_right]

                if self.noise_x:
                    # Randomly translate the input to the encoder to encourage
                    # translational invariance
                    total_len = coarse_f.size(1)
                    translated = []
                    for j in range(coarse_f.size(0)):
                        shift = random.randrange(256) - 128
                        translated.append(
                            noisy_f[j, pad_left - pad_left_encoder + shift:total_len - extra_pad_right + shift])
                    translated = torch.stack(translated, dim=0)
                else:
                    translated = noisy_f[:, pad_left - pad_left_encoder:]
                p_cf, vq_pen, encoder_pen, entropy = self(speaker, x, translated)
                p_c, p_f = p_cf
                loss_c = criterion(p_c.transpose(1, 2).float(), y_coarse)
                loss_f = criterion(p_f.transpose(1, 2).float(), y_fine)
                encoder_weight = 0.01 * min(1, max(0.1, step / 1000 - 1))
                loss = loss_c + loss_f + vq_pen + encoder_weight * encoder_pen

                optimiser.zero_grad()
                if use_half:
                    optimiser.backward(loss)
                    if do_clip:
                        raise RuntimeError("clipping in half precision is not implemented yet")
                else:
                    loss.backward()
                    if do_clip:
                        max_grad = 0
                        max_grad_name = ""
                        for name, param in self.named_parameters():
                            if param.grad is not None:
                                param_max_grad = param.grad.data.abs().max()
                                if param_max_grad > max_grad:
                                    max_grad = param_max_grad
                                    max_grad_name = name
                                if 1000000 < param_max_grad:
                                    logger.log(f'Very large gradient at {name}: {param_max_grad}')
                        if 100 < max_grad:
                            for param in self.parameters():
                                if param.grad is not None:
                                    if 1000000 < max_grad:
                                        param.grad.data.zero_()
                                    else:
                                        param.grad.data.mul_(100 / max_grad)
                        if running_max_grad < max_grad:
                            running_max_grad = max_grad
                            running_max_grad_name = max_grad_name

                        if 100000 < max_grad:
                            torch.save(self.state_dict(), "bad_model.pyt")
                            raise RuntimeError("Aborting due to crazy gradient (model saved to bad_model.pyt)")
                optimiser.step()
                running_loss_c += loss_c.item()
                running_loss_f += loss_f.item()
                running_loss_vq += vq_pen.item()
                running_loss_vqc += encoder_pen.item()
                running_entropy += entropy

                self.after_update()

                speed = (i + 1) / (time.time() - start)
                avg_loss_c = running_loss_c / (i + 1)
                avg_loss_f = running_loss_f / (i + 1)
                avg_loss_vq = running_loss_vq / (i + 1)
                avg_loss_vqc = running_loss_vqc / (i + 1)
                avg_entropy = running_entropy / (i + 1)

                step += 1
                k = step // 1000
                logger.status(
                    f'[Training] Epoch: {e + 1}/{epochs} -- Batch: {i + 1}/{iters} -- Loss: c={avg_loss_c:#.4} f={avg_loss_f:#.4} vq={avg_loss_vq:#.4} vqc={avg_loss_vqc:#.4} -- Entropy: {avg_entropy:#.4} -- Grad: {running_max_grad:#.1} {running_max_grad_name} Speed: {speed:#.4} steps/sec -- Step: {k}k ')

                # tensorboard writer
                writer.add_scalars('Train/loss_group', {'loss_c': loss_c.item(),
                                                        'loss_f': loss_f.item(),
                                                        'vq': vq_pen.item(),
                                                        'vqc': encoder_pen.item(),
                                                        'entropy': entropy, }, step - 1)

            os.makedirs(paths.checkpoint_dir, exist_ok=True)
            torch.save({'epoch': e,
                        'state_dict': self.state_dict(),
                        'optimiser': optimiser.state_dict(),
                        'step': step},
                       paths.model_path())
            # torch.save(self.state_dict(), paths.model_path())
            # np.save(paths.step_path(), step)
            logger.log_current_status()
            # logger.log(f' <saved>; w[0][0] = {self.overtone.wavernn.gru.weight_ih_l0[0][0]}')

            if e % test_epochs == 0:
                torch.save({'epoch': e,
                            'state_dict': self.state_dict(),
                            'optimiser': optimiser.state_dict(),
                            'step': step},
                           paths.model_hist_path(step))
                self.do_test(writer, e, step, dataset.path, valid_index)
                self.do_test_generate(paths, step, dataset.path, valid_index)
            # if k > saved_k + 50:
            #     torch.save({'epoch': e,
            #                 'state_dict': self.state_dict(),
            #                 'optimiser': optimiser.state_dict(),
            #                 'step': step},
            #                paths.model_hist_path(step))
            #     # torch.save(self.state_dict(), paths.model_hist_path(step))
            #     saved_k = k
            #     self.do_generate(paths, step, dataset.path, valid_index)

    def do_test(self, writer, epoch, step, data_path, test_index):
        dataset = env.MultispeakerDataset(test_index, data_path)
        criterion = nn.NLLLoss().cuda()
        # k = 0
        # saved_k = 0
        pad_left = self.pad_left()
        pad_left_encoder = self.pad_left_encoder()
        pad_left_decoder = self.pad_left_decoder()
        extra_pad_right = 0
        pad_right = self.pad_right() + extra_pad_right
        window = 16 * self.total_scale()

        test_loader = DataLoader(dataset,
                                 collate_fn=lambda batch: env.collate_multispeaker_samples(pad_left, window, pad_right,
                                                                                           batch),
                                 batch_size=16, num_workers=2, shuffle=False, pin_memory=True)

        running_loss_c = 0.
        running_loss_f = 0.
        running_loss_vq = 0.
        running_loss_vqc = 0.
        running_entropy = 0.
        running_max_grad = 0.
        running_max_grad_name = ""

        for i, (speaker, wave16) in enumerate(test_loader):
            speaker = speaker.cuda()
            wave16 = wave16.cuda()

            coarse = (wave16 + 2 ** 15) // 256
            fine = (wave16 + 2 ** 15) % 256

            coarse_f = coarse.float() / 127.5 - 1.
            fine_f = fine.float() / 127.5 - 1.
            total_f = (wave16.float() + 0.5) / 32767.5

            noisy_f = total_f

            x = torch.cat([
                coarse_f[:, pad_left - pad_left_decoder:-pad_right].unsqueeze(-1),
                fine_f[:, pad_left - pad_left_decoder:-pad_right].unsqueeze(-1),
                coarse_f[:, pad_left - pad_left_decoder + 1:1 - pad_right].unsqueeze(-1),
            ], dim=2)
            y_coarse = coarse[:, pad_left + 1:1 - pad_right]
            y_fine = fine[:, pad_left + 1:1 - pad_right]

            translated = noisy_f[:, pad_left - pad_left_encoder:]

            p_cf, vq_pen, encoder_pen, entropy = self(speaker, x, translated)
            p_c, p_f = p_cf
            loss_c = criterion(p_c.transpose(1, 2).float(), y_coarse)
            loss_f = criterion(p_f.transpose(1, 2).float(), y_fine)
            # encoder_weight = 0.01 * min(1, max(0.1, step / 1000 - 1))
            # loss = loss_c + loss_f + vq_pen + encoder_weight * encoder_pen

            running_loss_c += loss_c.item()
            running_loss_f += loss_f.item()
            running_loss_vq += vq_pen.item()
            running_loss_vqc += encoder_pen.item()
            running_entropy += entropy

        avg_loss_c = running_loss_c / (i + 1)
        avg_loss_f = running_loss_f / (i + 1)
        avg_loss_vq = running_loss_vq / (i + 1)
        avg_loss_vqc = running_loss_vqc / (i + 1)
        avg_entropy = running_entropy / (i + 1)

        k = step // 1000
        logger.log(
            f'[Testing] Epoch: {epoch} -- Loss: c={avg_loss_c:#.4} f={avg_loss_f:#.4} vq={avg_loss_vq:#.4} vqc={avg_loss_vqc:#.4} -- Entropy: {avg_entropy:#.4} -- Grad: {running_max_grad:#.1} {running_max_grad_name} -- Step: {k}k ')

        # tensorboard writer
        writer.add_scalars('Test/loss_group', {'loss_c': avg_loss_c,
                                               'loss_f': avg_loss_f,
                                               'vq': avg_loss_vq,
                                               'vqc': avg_loss_vqc,
                                               'entropy': avg_entropy, }, step - 1)

    def do_test_generate(self, paths, step, data_path, test_index, deterministic=False, use_half=False, verbose=False):
        k = step // 1000
        test_index = [x[:2] if len(x) > 0 else [] for i, x in enumerate(test_index)]
        dataset = env.MultispeakerDataset(test_index, data_path)
        loader = DataLoader(dataset, shuffle=False)
        data = [x for x in loader]
        n_points = len(data)
        gt = [(x[0].float() + 0.5) / (2 ** 15 - 0.5) for speaker, x in data]
        extended = [np.concatenate(
            [np.zeros(self.pad_left_encoder(), dtype=np.float32), x, np.zeros(self.pad_right(), dtype=np.float32)]) for
                    x in gt]
        speakers = [torch.FloatTensor(speaker[0].float()) for speaker, x in data]
        maxlen = max([len(x) for x in extended])
        aligned = [torch.cat([torch.FloatTensor(x), torch.zeros(maxlen - len(x))]) for x in extended]
        os.makedirs(paths.gen_path(), exist_ok=True)
        out, _, _, _ = self.forward_generate(torch.stack(speakers + list(reversed(speakers)), dim=0).cuda(),
                                       torch.stack(aligned + aligned, dim=0).cuda(), verbose=verbose, use_half=use_half)

        logger.log(f'out: {out.size()}')
        for i, x in enumerate(gt):
            librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_target.wav', x.cpu().numpy(), sr=sample_rate)
            audio = out[i][:len(x)].cpu().numpy()
            librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_generated.wav', audio, sr=sample_rate)
            audio_tr = out[n_points + i][:len(x)].cpu().numpy()
            librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_transferred.wav', audio_tr, sr=sample_rate)

    def do_generate(self, 
                    paths,
                    data_path,
                    index,
                    test_speakers,
                    test_utts_per_speaker,
                    use_half=False,
                    verbose=False, 
                    only_discrete=False):

        # Set the speaker to generate for each utterance
        # speaker_id = 1  # the speaker id to condition the model on for generation # TODO make this a CLA?

        # Get the utts we have chosen to generate from 'index'
        # 'index' contains ALL utts in dataset
        test_index = []
        for i, x in enumerate(index):
            if test_speakers == 0 or i < test_speakers:
                if test_utts_per_speaker == 0:
                    # if test_utts_per_speaker is 0, then use ALL utts for the speaker
                    test_index.append(x)
                else:
                    test_index.append(x[:test_utts_per_speaker])
            else:
                test_index.append([])  # done so that speaker one hots are created of correct dimension


        # test_index = [x[:test_utts_per_speaker] if len(x) > 0 else [] for i, x in enumerate(test_index)]

        # logger.log('second:')
        # logger.log(test_index)

        # make containing directories
        os.makedirs(f'{paths.gen_path()}embeddings', exist_ok=True)
        os.makedirs(f'{paths.gen_path()}vqvae_tokens', exist_ok=True)
        os.makedirs(f'{paths.gen_path()}decoder_input_vectors', exist_ok=True)

        # TODO Save embedding matrix to disk for plotting and analysis
        torch.save(self.vq.embedding0.clone().detach(), f'{paths.gen_path()}embeddings/vqvae_codebook.pt')

        dataset = env.MultispeakerDataset(test_index, data_path, return_filename=True)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        for speaker, x, filename in loader:  # NB!!! Following code in for loop is only designed for batch size == 1 for now

            print("speaker.size()", speaker.size())
            print("x.size()", x.size())
            print("filename", filename)

            # data = [x for x in loader]

            # logger.log("data:")
            # logger.log(f"len(data) = {len(data)}")
            # logger.log(f"data[0]: {data[0]}")

            # n_points = len(data)
            # gt = [(x[0].float() + 0.5) / (2 ** 15 - 0.5) for speaker, x, filename in data]
            # extended = [np.concatenate(
            #     [np.zeros(self.pad_left_encoder(), dtype=np.float32), x, np.zeros(self.pad_right(), dtype=np.float32)]) for
            #             x in gt]

            gt = (x[0].float() + 0.5) / (2 ** 15 - 0.5)
            extended = np.concatenate([np.zeros(self.pad_left_encoder(), dtype=np.float32), gt, np.zeros(self.pad_right(), dtype=np.float32)])

            # TODO use speaker id from dataset
            speakers = [torch.FloatTensor(speaker[0].float())] # TODO seems to only have 3 speakers? As per the CLA. look at dataset...

            total_test_utts = test_speakers * test_utts_per_speaker
            print("test_speakers", test_speakers)
            print("test_utts_per_speaker", test_utts_per_speaker)

            # (np.arange(30) == 1) is a one hot conditioning vector indicating speaker 2
            # vc_speakers = [torch.FloatTensor((np.arange(30) == speaker_id).astype(np.float)) for _ in range(total_test_utts)]
            # speakers = vc_speakers

            print("speakers:")
            print("speakers", speakers)
            print("len(speakers)", len(speakers))
            print("speakers[0].size()", speakers[0].size())
            print("torch.stack(speakers, dim=0).size()", torch.stack(speakers, dim=0).size())

            # maxlen = max([len(x) for x in extended])
            print("extended.shape", extended.shape)
            maxlen = len(extended)

            # aligned = [torch.cat([torch.FloatTensor(x), torch.zeros(maxlen - len(x))]) for x in extended]
            aligned = [torch.FloatTensor(extended)]
            print("torch.stack(aligned, dim=0).size()",torch.stack(aligned, dim=0).size())



            # out = self.forward_generate(torch.stack(speakers + list(reversed(speakers)), dim=0).cuda(), torch.stack(aligned + aligned, dim=0).cuda(), verbose=verbose, use_half=use_half, only_discrete=only_discrete)
            out, discrete, index_atom, index_group = self.forward_generate(torch.stack(speakers, dim=0).cuda(),
                                                  torch.stack(aligned, dim=0).cuda(), verbose=verbose, use_half=use_half,
                                                  only_discrete=only_discrete)

            if out is not None:
                logger.log(f'out[0]: {out[0]}')
                logger.log(f'out: {out.size()}')
            logger.log(f'index_atom.size(): {index_atom.size()}')
            # logger.log(f'index_atom[0]: {index_atom[0]}')
            logger.log(f'index_atom[0].size(): {index_atom[0].size()}')
            logger.log(f'index_group.size(): {index_group.size()}')
            # logger.log(f'index_group[0]: {index_group[0]}')
            logger.log(f'index_group[0].size(): {index_group[0].size()}')

            # for i, x in enumerate(gt) :
            #     librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_target.wav', x.cpu().numpy(), sr=sample_rate)
            #     audio = out[i][:len(x)].cpu().numpy()
            #     librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_generated.wav', audio, sr=sample_rate)
            #     audio_tr = out[n_points+i][:len(x)].cpu().numpy()
            #     librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_transferred.wav', audio_tr, sr=sample_rate)

            ######################################
            # Generate atom and group data to save to disk
            index_atom = index_atom.squeeze()
            index_group = index_group.squeeze()
            assert index_atom.size() == index_group.size()
            vqvae_tokens = []
            for i in range(len(index_atom)):
                atom_id = int(index_atom[i])
                group_id = int(index_group[i])
                vqvae_tokens.append(f"{group_id}_{atom_id}")
            vqvae_tokens = '\n'.join(vqvae_tokens)

            ######################################
            # Save files to disk

            # Discrete vqvae symbols
            # for i, x in enumerate(gt):
            # os.makedirs(f'{paths.gen_path()}groups', exist_ok=True)
            filename_noext = f'{filename[0]}'
            with open(f'{paths.gen_path()}vqvae_tokens/{filename_noext}.txt','w') as f:
                f.write(vqvae_tokens)

            # TODO The ACTUAL embeddings fed into the decoder
            # TODO (average of atoms in group weighted according to their distance from encoder output)
            torch.save(discrete, f'{paths.gen_path()}decoder_input_vectors/{filename_noext}.pt')

            # discrete vqvae tokens for analysis and modification/pronunciation correction
            # torch.save(index_atom, f'{paths.gen_path()}atoms/{filename_noext}_atom.pt')
            # torch.save(index_group, f'{paths.gen_path()}groups/{filename_noext}_group.pt')
            # TODO currently we are saving the entire matrix of discrete tokens for all utts multiple times
            # TODO need to change this so that we are saving a single vector of discrete tokens for each input test utt
            # TODO create more informative filenames for test generated utts. use original vctk filename and include the speaker that was used to condition the model (create a mapping from one hot speaker id [0-30] to vctk speaker names [pxxx-pzzz] to do this)

            # print(len(index_atom.tolist()))
            # print(len(index_group.tolist()))
            # print(index_atom.tolist())
            # print(index_group.tolist())

            # save wav file for listening
            if out is not None:
                audio_tr = out[0][:self.pad_left_encoder() + len(gt)].cpu().numpy()
                wav_path = f'{paths.gen_path()}{filename_noext}.wav'
                librosa.output.write_wav(wav_path, audio_tr, sr=sample_rate)
                print(f"Saved audio to {wav_path}")

    def do_generate_from_tokens(self,
                    paths,
                    tokens_path,
                    verbose=False):
        # Set the speaker to generate for each utterance
        # speaker_id = 1  # the speaker id to condition the model on for generation # TODO make this a CLA?

        # Get the utts we have chosen to generate from 'index'
        # # 'index' contains ALL utts in dataset
        # test_index = []
        # for i, x in enumerate(index):
        #     if test_speakers == 0 or i < test_speakers:
        #         if test_utts_per_speaker == 0:
        #             # if test_utts_per_speaker is 0, then use ALL utts for the speaker
        #             test_index.append(x)
        #         else:
        #             test_index.append(x[:test_utts_per_speaker])
        #     else:
        #         test_index.append([])  # done so that speaker one hots are created of correct dimension


        # test_index = [x[:test_utts_per_speaker] if len(x) > 0 else [] for i, x in enumerate(test_index)]

        # logger.log('second:')
        # logger.log(test_index)

        # # make containing directories
        # os.makedirs(f'{paths.gen_path()}embeddings', exist_ok=True)
        # os.makedirs(f'{paths.gen_path()}vqvae_tokens', exist_ok=True)
        #
        # # TODO Save embedding matrix to disk for plotting and analysis
        # torch.save(self.vq.embedding0.clone().detach(), f'{paths.gen_path()}embeddings/vqvae_codebook.pt')
        #
        # dataset = env.MultispeakerDataset(test_index, data_path, return_filename=True)
        # loader = DataLoader(dataset, batch_size=1, shuffle=False)

        # for speaker, x, filename in loader:  # NB!!! Following code in for loop is only designed for batch size == 1 for now

        # print("speaker.size()", speaker.size())
        # print("x.size()", x.size())
        # print("filename", filename)
        #
        # # data = [x for x in loader]
        #
        # # logger.log("data:")
        # # logger.log(f"len(data) = {len(data)}")
        # # logger.log(f"data[0]: {data[0]}")
        #
        # # n_points = len(data)
        # # gt = [(x[0].float() + 0.5) / (2 ** 15 - 0.5) for speaker, x, filename in data]
        # # extended = [np.concatenate(
        # #     [np.zeros(self.pad_left_encoder(), dtype=np.float32), x, np.zeros(self.pad_right(), dtype=np.float32)]) for
        # #             x in gt]
        #
        # gt = (x[0].float() + 0.5) / (2 ** 15 - 0.5)
        # extended = np.concatenate([np.zeros(self.pad_left_encoder(), dtype=np.float32), gt, np.zeros(self.pad_right(), dtype=np.float32)])

        ##########################################################
        # Load tokens from file @ tokens_path
        with open(tokens_path, 'r') as f:
            tokens = f.readlines()
        # print(tokens)
        groups = [int(line.split('_')[0]) for line in tokens]
        # print(groups)

        ##########################################################
        # TODO Get speaker id from filename
        speaker_id = 0
        total_test_utts = 1
        total_test_speakers = 30
        speakers = [torch.FloatTensor((np.arange(total_test_speakers) == speaker_id).astype(np.float)) for _ in range(total_test_utts)]

        ##########################################################
        # Get embeddings corresponding to groups
        discretes = []
        groups_tensor = torch.zeros(0)
        num_atoms_per_group = 10
        for g in groups:
            # get the embeddings corresponding to this group from the the atoms codebook
            # print(self.vq.embedding0.size())  # torch.Size([1, 410, 128])
            group_embeddings = self.vq.embedding0[:, g*num_atoms_per_group:(g+1)*num_atoms_per_group, :]

            # get the averaged embedding
            discrete = torch.mean(group_embeddings, dim=1)
            # print(discrete.size())
            # TODO correctly weight the atoms according to their distance from the group centre, equation 6
            discretes.append(discrete)

        discretes = torch.cat(discretes, dim=0).cuda()

        # print(discretes.size())

        # total_test_utts = test_speakers * test_utts_per_speaker
        # print("test_speakers", test_speakers)
        # print("test_utts_per_speaker", test_utts_per_speaker)

        # (np.arange(30) == 1) is a one hot conditioning vector indicating speaker 2
        # vc_speakers = [torch.FloatTensor((np.arange(30) == speaker_id).astype(np.float)) for _ in range(total_test_utts)]
        # speakers = vc_speakers

        # print("speakers:")
        # print("speakers", speakers)
        # print("len(speakers)", len(speakers))
        # print("speakers[0].size()", speakers[0].size())
        # print("torch.stack(speakers, dim=0).size()", torch.stack(speakers, dim=0).size())
        #
        # # maxlen = max([len(x) for x in extended])
        # print("extended.shape", extended.shape)
        # maxlen = len(extended)
        #
        # # aligned = [torch.cat([torch.FloatTensor(x), torch.zeros(maxlen - len(x))]) for x in extended]
        # aligned = [torch.FloatTensor(extended)]
        # print("torch.stack(aligned, dim=0).size()",torch.stack(aligned, dim=0).size())



        # out = self.forward_generate(torch.stack(speakers + list(reversed(speakers)), dim=0).cuda(), torch.stack(aligned + aligned, dim=0).cuda(), verbose=verbose, use_half=use_half, only_discrete=only_discrete)
        # out, index_atom, index_group = self.forward_generate(torch.stack(speakers, dim=0).cuda(),
        #                                       torch.stack(aligned, dim=0).cuda(), verbose=verbose, use_half=use_half,
        #                                       only_discrete=only_discrete)

        out = self.forward_generate_from_tokens(
            torch.stack(speakers, dim=0).cuda(),
            discretes,
            verbose=verbose,
        )

        # if out is not None:
        #     logger.log(f'out[0]: {out[0]}')
        #     logger.log(f'out: {out.size()}')
        # logger.log(f'index_atom.size(): {index_atom.size()}')
        # # logger.log(f'index_atom[0]: {index_atom[0]}')
        # logger.log(f'index_atom[0].size(): {index_atom[0].size()}')
        # logger.log(f'index_group.size(): {index_group.size()}')
        # # logger.log(f'index_group[0]: {index_group[0]}')
        # logger.log(f'index_group[0].size(): {index_group[0].size()}')

        # for i, x in enumerate(gt) :
        #     librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_target.wav', x.cpu().numpy(), sr=sample_rate)
        #     audio = out[i][:len(x)].cpu().numpy()
        #     librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_generated.wav', audio, sr=sample_rate)
        #     audio_tr = out[n_points+i][:len(x)].cpu().numpy()
        #     librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_transferred.wav', audio_tr, sr=sample_rate)

        # ######################################
        # # Generate atom and group data to save to disk
        # index_atom = index_atom.squeeze()
        # index_group = index_group.squeeze()
        # assert index_atom.size() == index_group.size()
        # vqvae_tokens = []
        # for i in range(len(index_atom)):
        #     atom_id = int(index_atom[i])
        #     group_id = int(index_group[i])
        #     vqvae_tokens.append(f"{group_id}_{atom_id}")
        # vqvae_tokens = '\n'.join(vqvae_tokens)
        #
        # ######################################
        # # Save files to disk
        # # for i, x in enumerate(gt):
        # # os.makedirs(f'{paths.gen_path()}groups', exist_ok=True)

        filename_noext = f'{os.path.basename(tokens_path).rstrip(".txt")}_from_tokens'
        # with open(f'{paths.gen_path()}vqvae_tokens/{filename_noext}.txt','w') as f:
        #     f.write(vqvae_tokens)

        # discrete vqvae tokens for analysis and modification/pronunciation correction
        # torch.save(index_atom, f'{paths.gen_path()}atoms/{filename_noext}_atom.pt')
        # torch.save(index_group, f'{paths.gen_path()}groups/{filename_noext}_group.pt')
        # TODO currently we are saving the entire matrix of discrete tokens for all utts multiple times
        # TODO need to change this so that we are saving a single vector of discrete tokens for each input test utt
        # TODO create more informative filenames for test generated utts. use original vctk filename and include the speaker that was used to condition the model (create a mapping from one hot speaker id [0-30] to vctk speaker names [pxxx-pzzz] to do this)

        # print(len(index_atom.tolist()))
        # print(len(index_group.tolist()))
        # print(index_atom.tolist())
        # print(index_group.tolist())

        # save wav file for listening
        if out is not None:
            audio_tr = out[0][:].cpu().numpy()
            # audio_tr = out[0][:self.pad_left_encoder() + len(gt)].cpu().numpy()
            wav_path = f'{paths.gen_path()}{filename_noext}.wav'
            librosa.output.write_wav(wav_path, audio_tr, sr=sample_rate)
            print(f"\nSaved audio to {wav_path}")
