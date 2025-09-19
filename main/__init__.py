from .pipelines.standard import get_cbm_standard as modular_get_cbm_standard
from .pipelines.joint import get_cbm_joint as modular_get_cbm_joint
from .pipelines.llm_mix_joint import get_cbm_LLM_mix_joint as modular_get_cbm_LLM_mix_joint


def get_cbm_standard(*args, **kwargs):
    return modular_get_cbm_standard(*args, **kwargs)


def get_cbm_joint(*args, **kwargs):
    return modular_get_cbm_joint(*args, **kwargs)


def get_cbm_LLM_mix_joint(*args, **kwargs):
    return modular_get_cbm_LLM_mix_joint(*args, **kwargs)
