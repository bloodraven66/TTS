import os
import sys
from huggingface_hub import hf_hub_download

class INFER():
    def __init__(self,
                lang,
                text,
                model='glowtts',
                vocoder='wg'
        ):
                
        self.SUPPORTED_LANGUAGES = ['mr']
        self.SUPPORTED_MODELS = ['glowtts']
        self.SUPPORTED_VOCODERS = ['wg', 'gl']

        self.lang = lang.lower()
        self.text = text
        self.model = model.lower()
        self.vocoder = vocoder.lower()
        self.load_model()
    
    def load_model(self):
        assert self.lang in self.SUPPORTED_LANGUAGES, 'language not supported'
        assert self.model in self.SUPPORTED_MODELS, 'model not supported'
        assert self.vocoder in self.SUPPORTED_VOCODERS, 'vocoder not supported'
        if self.model == 'glowtts':
            from glowtts.infer_glowtts import inference
            sys.path.append('TTS/glowtts')
            self.inference = inference
            self.download_checkpoint()
    
    def download_checkpoint(self):
        checkpoint_path = f'./chk/{self.model}/{self.lang}'
        if not os.path.exists(checkpoint_path):
            if not os.path.exists('./chk'):
                os.mkdir('./chk')
            if not os.path.exists('./chk/{self.model}'):
                os.mkdir('./chk/{self.model}')
            os.mkdir(checkpoint_path)
        if not os.path.exists(os.path.join(checkpoint_path, f'{self.lang}.pth')):
            hf_hub_download(
                            repo_id='SYSPIN/Marathi_Male_GlowTTS_waveglow',
                            filename='G_434.pth',
                            cache_dir=checkpoint_path,
                            force_filename='mr.pth'
                            )
        if not os.path.exists(os.path.join(checkpoint_path, 'config.json')):
            hf_hub_download(
                            repo_id='SYSPIN/Marathi_Male_GlowTTS_waveglow',
                            filename='config.json',
                            cache_dir=checkpoint_path,
                            force_filename='config.json'
                            )
            

    
    def run(self):
        self.inference(vocoder=self.vocoder,
                        text=self.text,
                        lang=self.lang,
                        vocoder_path=None
                        )

infer = INFER(lang='mr', text='हा मराठी मजकूर आहे', vocoder='gl')
infer.run()
