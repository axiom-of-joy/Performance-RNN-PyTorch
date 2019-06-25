import torch
from config import device, model as model_config
from model import PerformanceRNN
import distiller


def main():
    sess_path = "save/ecomp.sess"
    state = torch.load(sess_path)
    model = PerformanceRNN(**state['model_config']).to(device)
    model.load_state_dict(state['model_state'])
    print("Done.")
    quantizer = distiller.quantization.PostTrainLinearQuantizer(
        model,
        bits_activations=8,
        bits_parameters=8,
        bits_accum=32,
        #overrides=None,
        mode=distiller.quantization.LinearQuantMode.SYMMETRIC,
        #clip_acts=ClipMode.NONE,
        #per_channel_wts=False,
        model_activation_stats=None,
        #fp16=False,
        #clip_n_stds=None,
        #scale_approx_mult_bits=None
    )
    quantizer.prepare_model()

if __name__ == "__main__":
    main()

