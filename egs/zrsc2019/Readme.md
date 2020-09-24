Code used to in part implement Interspeech 2019 paper:

@inproceedings{Feng2019,
  author={Siyuan Feng and Tan Lee and Zhiyuan Peng},
  title={{Combining Adversarial Training and Disentangled Speech Representation for Robust Zero-Resource Subword Modeling}},
  year=2019,
  booktitle={Proc. Interspeech 2019},
  pages={1093--1097},
  doi={10.21437/Interspeech.2019-1337},
  url={http://dx.doi.org/10.21437/Interspeech.2019-1337}
}

This recipe is based on Wei-Ning Hsu's FHVAE implementation [1].


[1] W. Hsu, Y. Zhang, and J. R. Glass, “Unsupervised learning of disentangled and interpretable representations from sequential data,”
in Proc. NIPS, 2017, pp. 1876–1887.


Usage:
To train an FHVAE using ZeroSpeech 2019 development database, run ./run_mfcc_novad_w_cm_fhvae_spkid_10_10.sh
To train using ZeroSpeech 2019 surprise database, run ./run_surprise_mfcc_novad_w_cm_fhvae_spkid_10_10.sh
