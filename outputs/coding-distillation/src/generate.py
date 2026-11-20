import json
import urllib.request
from urllib.parse import urlparse


class Ollama:
    def __init__(self, config):
        self.config = config
        url = config['base_url']
        if urlparse(url).hostname not in ('localhost', '127.0.0.1', '::1'):
            raise ValueError('This pipeline only sends prompts to local Ollama')
        self.base = url.rstrip('/')
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def api(self, path, payload=None, timeout=None):
        req = urllib.request.Request(self.base + path,
              data=json.dumps(payload).encode() if payload is not None else None,
              headers={'Content-Type': 'application/json'})
        with self.opener.open(req, timeout=timeout or self.config['timeout_seconds']) as response:
            result = json.load(response)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def options_for(self, model):
        opts = dict(self.config.get('options', {}))
        for prefix, overrides in self.config.get('model_options', {}).items():
            if model == prefix or model.startswith(prefix):
                opts.update({k: v for k, v in overrides.items() if k != 'timeout_seconds'})
        return opts

    def timeout_for(self, model):
        for prefix, overrides in self.config.get('model_options', {}).items():
            if (model == prefix or model.startswith(prefix)) and 'timeout_seconds' in overrides:
                return overrides['timeout_seconds']
        return self.config['timeout_seconds']

    def metadata(self, model):
        tags = self.api('/api/tags')['models']
        entry = next((t for t in tags if t['name'] == model), None)
        if entry is None:
            raise ValueError(f'Missing exact teacher tag: {model}; no quantized substitution is allowed')
        if entry['details'].get('quantization_level') not in ('F16', 'FP16'):
            raise ValueError(f'{model} is not FP16')
        return {'tag': entry, 'show': self.api('/api/show', {'model': model}),
                'version': self.api('/api/version')}

    def generate(self, model, prompt):
        return self.api('/api/chat', {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
                        'stream': False, 'keep_alive': self.config['keep_alive'],
                        'options': self.options_for(model)}, timeout=self.timeout_for(model))

    def unload(self, model):
        self.api('/api/generate', {'model': model, 'keep_alive': 0, 'stream': False})

    def unload_all(self):
        for item in self.api('/api/ps').get('models', []):
            self.unload(item['name'])
