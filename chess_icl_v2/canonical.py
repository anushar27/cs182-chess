"""Cohort-independent frozen inference: one unpadded episode, immutable-input cache."""
from pathlib import Path
from .common import digest,read_json,write_json
from .metrics import validate_probabilities


class CanonicalCache:
    def __init__(self,predictor,fingerprint,path):
        self.predictor=predictor;self.fingerprint=fingerprint;self.path=Path(path)
        self.entries={};self.new_predictions=0
        if self.path.exists():
            data=read_json(self.path)
            if data.get('format')!='canonical_unpadded_batch1_v1':raise ValueError('Unsupported canonical cache format')
            if data['fingerprint']!=fingerprint:raise ValueError('Canonical cache model/backend mismatch')
            self.entries=data['entries']
            for probability in self.entries.values():validate_probabilities(probability)

    def predict_proba(self,episode):
        key=digest(episode.input_ids)
        if key not in self.entries:
            probability=self.predictor.predict_proba(episode)
            validate_probabilities(probability)
            self.entries[key]=probability;self.new_predictions+=1
        return list(self.entries[key])

    def save(self):
        write_json(self.path,{'format':'canonical_unpadded_batch1_v1','fingerprint':self.fingerprint,'entries':self.entries})
