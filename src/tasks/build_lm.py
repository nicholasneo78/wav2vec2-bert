import subprocess
import logging
import os
from typing import Dict

from args_manager import BuildLMArgs, ClearMLBuildLMArgs

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
    datefmt='%H:%M:%S'
)

class BuildLM:

    '''
        get the text file and produce a kenlm arpa file
    '''

    def __init__(
        self, 
        cfg: Dict,
        is_remote: bool,
    ) -> None:
        
        args = cfg.build_lm

        if not is_remote:
            self.config_dict = {
                'n_grams': args.n_grams,
                'kenlm_path': args.kenlm_path,
                'text_file_path': args.text_file_path,
                'raw_output_lm_path': args.raw_output_lm_path,
                'corrected_output_lm_path': args.corrected_output_lm_path,
            }
        else:
            self.config_dict = {
                'n_grams': args.n_grams,
                'kenlm_path': args.kenlm_path,
                'text_file_path': os.path.join(args.temp.text_file_path, args.data.input_text_file_path),
                'raw_output_lm_path': args.raw_output_lm_path,
                'corrected_output_lm_path': args.corrected_output_lm_path,
            }

    def build_lm(self) -> str:
        
        '''
        process to build the kenlm language model arpa file
        '''

        logging.getLogger('INFO').info("Building the LM...")

        subprocess.run(["chmod", "-R", "777", self.config_dict['kenlm_path']])
        command = f"{os.path.join(self.config_dict['kenlm_path'], 'build/bin/lmplz')} -o {self.config_dict['n_grams']} < {self.config_dict['text_file_path']} > {self.config_dict['raw_output_lm_path']} --discount_fallback"
        subprocess.run(command, shell=True) 

        self.fix_lm_path()

        # returns the file path of the generated kenlm language model arpa file
        return self.config_dict['corrected_output_lm_path']
    
    def fix_lm_path(self) -> None:

        '''
        to fix the eos bug in the raw kenlm LM file and output the corrected LM
        '''
        
        with open(self.config_dict['raw_output_lm_path'], "r") as read_file, open(self.config_dict['corrected_output_lm_path'], "w") as write_file:
            has_added_eos = False
            for line in read_file:
                if not has_added_eos and "ngram 1=" in line:
                    count=line.strip().split("=")[-1]
                    write_file.write(line.replace(f"{count}", f"{int(count)+1}"))
                elif not has_added_eos and "<s>" in line:
                    write_file.write(line)
                    write_file.write(line.replace("<s>", "</s>"))
                    has_added_eos = True
                else:
                    write_file.write(line)

        return self.config_dict['raw_output_lm_path'], self.config_dict['corrected_output_lm_path']
                
    def __call__(self):
        return self.build_lm()
