from __future__ import absolute_import
from ctypes import *
from .bytes_wrapper import WrappedDataPtr

import os
import platform


path = f"{os.path.dirname(os.path.abspath(__file__))}/../env/secret/"

if platform.system() == "Windows":
    SECRET_ENVS_DYN_LIB_PATHS = f"{path}windows_secret.dll"
elif platform.system() == "Darwin":
    SECRET_ENVS_DYN_LIB_PATHS = f"{path}macos_secret.dylib"
elif platform.system() == "Linux":
    SECRET_ENVS_DYN_LIB_PATHS = f"{path}linux_secret.so"
else:
    raise Exception("PLATFORM NOT RECOGNIZED !")

_dll = cdll.LoadLibrary(SECRET_ENVS_DYN_LIB_PATHS)

_dll.free_wrapped_bytes.argtypes = [WrappedDataPtr]
_dll.free_wrapped_bytes.restype = None

_dll.create_secret_env1.argtypes = []
_dll.create_secret_env1.restype = c_void_p

_dll.get_mdp_env_data.argtypes = [c_void_p]
_dll.get_mdp_env_data.restype = WrappedDataPtr

_dll.mdp_env_is_state_terminal.argtypes = [c_void_p, c_uint64]
_dll.mdp_env_is_state_terminal.restype = c_bool

_dll.mdp_env_transition_probability.argtypes = [c_void_p, c_uint64, c_uint64, c_uint64, c_float]
_dll.mdp_env_transition_probability.restype = c_float

_dll.delete_mdp_env.argtypes = [c_void_p]
_dll.delete_mdp_env.restype = None

_dll.free_wrapped_bytes.argtypes = [WrappedDataPtr]
_dll.free_wrapped_bytes.restype = None

_dll.create_secret_env2.argtypes = []
_dll.create_secret_env2.restype = c_void_p

_dll.create_secret_env3.argtypes = []
_dll.create_secret_env3.restype = c_void_p

_dll.act_on_single_agent_env.argtypes = [c_void_p, c_uint64]
_dll.act_on_single_agent_env.restype = None

_dll.get_single_agent_env_state_data.argtypes = [c_void_p]
_dll.get_single_agent_env_state_data.restype = WrappedDataPtr

_dll.reset_single_agent_env.argtypes = [c_void_p]
_dll.reset_single_agent_env.restype = None

_dll.reset_random_single_agent_env.argtypes = [c_void_p]
_dll.reset_random_single_agent_env.restype = None

_dll.delete_single_agent_env.argtypes = [c_void_p]
_dll.delete_single_agent_env.restype = None


def get_dll():
    global _dll
    return _dll
