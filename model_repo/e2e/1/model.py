import json
import logging
import numpy as np 
import subprocess
import sys

import triton_python_backend_utils as pb_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TritonPythonModel:
    """This model loops through different dtypes to make sure that
    serialize_byte_tensor works correctly in the Python backend.
    """
    
    def __mean_pooling(self, token_embeddings, attention_mask):
        logger.info("token_embeddings: {}".format(token_embeddings))
        logger.info("attention_mask: {}".format(attention_mask))
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    

    def initialize(self, args):
        self.model_dir = args['model_repository']
        
        # Workaround for SageMaker provided model repo dir ----------------
        # split_dir = self.model_dir.split(sep='/')[1:-2]
        # self.model_dir = ''
        # for word in split_dir:
             # self.model_dir += f'/{word}'
        # ------------------------------------------------ End of Workaround 
        
        subprocess.check_call([sys.executable, "-m", "pip", "install", '-r', f'{self.model_dir}/requirements.txt'])
        global transformers, torch
        import torch 
        import transformers
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(f'{self.model_dir}/tokenizer')
        self.device_id = args['model_instance_device_id']
        self.device = torch.device(f'cuda:{self.device_id}') if torch.cuda.is_available() else torch.device('cpu')
        self.model = transformers.AutoModel.from_pretrained(f'{self.model_dir}/model').eval().to(self.device)
        self.model_config = model_config = json.loads(args['model_config'])

        output0_config = pb_utils.get_output_config_by_name(
            model_config, "SENT_EMBED")

        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        
    def execute(self, requests):
        
        file = open("logs.txt", "w")
        
        responses = []
        for request in requests:
            logger.info("Request: {}".format(request))
            
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            in_0 = in_0.as_numpy()
            
            logger.info("in_0: {}".format(in_0))
                        
            tok_batch = []
            
            for i in range(in_0.shape[0]):                
                decoded_object = in_0[i,0].decode()
                
                logger.info("decoded_object: {}".format(decoded_object))
                                                
                tok_batch.append(decoded_object)
                
            logger.info("tok_batch: {}".format(tok_batch))
            
            tok_sent = self.tokenizer(tok_batch,
                                      padding='max_length',
                                      max_length=128,
                                      return_tensors='pt'
                                     )
            input_tuple = tok_sent['input_ids'].to(self.device), tok_sent['attention_mask'].to(self.device)
            
            with torch.no_grad():
                with torch.autocast(device_type='cuda',enabled=True):
                    model_output = self.model(*input_tuple)
                    tok_embeds = model_output[0]
            
                    sentence_embeddings = self.__mean_pooling(tok_embeds, input_tuple[1])
                    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

            out_0 = np.array(sentence_embeddings.cpu(),dtype=self.output0_dtype)
            logger.info("out_0: {}".format(out_0))
            
            out_tensor_0 = pb_utils.Tensor("SENT_EMBED", out_0)
            logger.info("out_tensor_0: {}".format(out_tensor_0))
            
            responses.append(pb_utils.InferenceResponse([out_tensor_0]))
            
        return responses
