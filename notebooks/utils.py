#
# utils.py
# Baysis
#
# Created by Vladimir Sotskov on 28/08/2021, 12:02.
# Copyright © 2021 Vladimir Sotskov. All rights reserved.
#

import h5py
import subprocess as sp
import numpy as np
from abc import abstractmethod, ABC
from enum import Enum, IntEnum, unique
from pathlib import Path

DATA_PATH = Path("../data")
RESULTS_PATH = Path("../data")
SMOOTHER_FNAME = "kalman_smoother_results.h5"


@unique
class SpecGroup(Enum):
    TRANSITION_MODEL = "transition"
    OBSERVATION_MODEL = "observation"
    DATA = "data"
    SAMPLER = "sampler"
    SIMULATION = "simulation"
    SMOOTHER = "smoother"


@unique
class ModelType(IntEnum):
    LINEAR_GAUSS = 1
    LINEAR_POISSON = 2
    BIMODAL_POISSON = 3

    def mean_function(self):
        if self == ModelType.BIMODAL_POISSON:
            return 4
        return 0

    def __str__(self):
        if self == ModelType.LINEAR_GAUSS:
            return "gauss"
        elif self == ModelType.LINEAR_POISSON:
            return "poiss1"
        elif self == ModelType.BIMODAL_POISSON:
            return "poiss2"


@unique
class DistributionType(IntEnum):
    NORMAL = 1
    UNIFORM = 2
    INVGAMMA = 3

    @property
    def nparams(self):
        return 2


@unique
class SamplerType(IntEnum):
    METROPOLIS = 1
    EHMM = 2

    def __str__(self):
        if self == SamplerType.METROPOLIS:
            return "met"
        if self == SamplerType.EHMM:
            return "ehmm"


@unique
class FilterType(IntEnum):
    COVARIANCE = 1
    INFORMATION = 2


@unique
class SmootherType(IntEnum):
    RTS = 1
    TWOFILTER = 2


@unique
class ParamType(IntEnum):
    DIAGONAL_MATRIX = 1
    SYMMETRIC_MATRIX = 2
    VECTOR = 3
    CONST_MATRIX = 4
    CONST_VECTOR = 5


class Specification:
    def __init__(self, specname: Enum, **kwargs):
        self._groupname_ = specname.value
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    def add2spec(self, spec):
        grp = spec.create_group(self._groupname_)
        for attr, value in vars(self).items():
            if attr is not None and attr[0] != "_":
                if isinstance(value, (list, np.ndarray)):
                    grp.create_dataset(attr, data=value)
                elif isinstance(value, IntEnum):
                    grp.attrs.create(attr, value.value, dtype=int)
                else:
                    grp.attrs.create(attr, value, dtype=h5py.string_dtype() if isinstance(value, str) else type(value))


class ModelParameter:
    def __init__(self, ptype: ParamType):
        self._type_ = ptype.value
        self.parametrized = False
        self.dim = None
        self._value_ = None

    def parametrize(self, dim: int, distribution: DistributionType, prior: tuple, minx=None, maxx=None, *args):
        self.parametrized = True
        self.dim = dim
        param_id = self._type_ + distribution.value * 10
        self._value_ = [param_id, 0 if minx is None else minx, 0 if maxx is None else maxx] + list(prior) + list(args)

    @abstractmethod
    @property
    def value(self):
        if self.parametrized:
            return np.array(self._value_)
        else:
            return self._value_

    @value.setter
    def value(self, val):
        if self.parametrized:
            raise ValueError("Can't set value as the matrix/vector is parametrised.")
        self._value_ = val


class SymmetricMatrixParam(ModelParameter):
    def __init__(self):
        super().__init__(ParamType.SYMMETRIC_MATRIX)

    @property
    def value(self):
        return super().value()


class DiagonalMatrixParam(ModelParameter):
    def __init__(self):
        super().__init__(ParamType.DIAGONAL_MATRIX)

    @property
    def value(self):
        return super().value()


class VectorParam(ModelParameter):
    def __init__(self):
        super().__init__(ParamType.VECTOR)

    @property
    def value(self):
        if self.parametrized:
            return np.array(self._value_)
        else:
            return self._value_.reshape(-1, 1)


class ConstParam(ModelParameter):
    def __init__(self, ptype: ParamType):
        super().__init__(ptype)

    @property
    def value(self):
        if not self.parametrized and self._type_ == ParamType.CONST_VECTOR:
            return self._value_.reshape(-1, 1)
        else:
            return super().value()


class TransitionSpec(Specification):
    def __init__(self, mean_coeff: ModelParameter, cov: ModelParameter, prior_mean: np.ndarray, prior_cov: np.ndarray):
        super().__init__(SpecGroup.TRANSITION_MODEL, A=mean_coeff.value, Q=cov.value, xdim=mean_coeff.dim,
                         mu_prior=prior_mean.reshape(-1, 1), S_prior=prior_cov)
        self.parametrised = mean_coeff.parametrized | cov.parametrized


class ObservationSpec(Specification):
    def __init__(self, model_type: ModelType, mean_coeff: ModelParameter, *params: ModelParameter):
        if model_type == ModelType.LINEAR_GAUSS:
            super().__init__(SpecGroup.OBSERVATION_MODEL, mtype=model_type,
                             C=mean_coeff.value, R=params[0].value, ydim=mean_coeff.dim)
        elif model_type == ModelType.LINEAR_POISSON:
            super().__init__(SpecGroup.OBSERVATION_MODEL, mtype=model_type,
                             C=mean_coeff.value, D=params[0].value, controls=params[1].value, ydim=mean_coeff.dim)
        elif model_type == ModelType.BIMODAL_POISSON:
            super().__init__(SpecGroup.OBSERVATION_MODEL, mtype=model_type,
                             C=mean_coeff.value, mean_function=model_type.mean_function(), ydim=mean_coeff.dim)
        self.parametrised = any([mean_coeff]+list(params))


class SamplerSpec(Specification):
    def __init__(self, sampler_type: SamplerType, pool_size: int = None, flip: bool = None):
        if sampler_type == SamplerType.METROPOLIS:
            super().__init__(SpecGroup.SAMPLER, stype=sampler_type)
        else:
            kwargs = dict(stype=sampler_type)
            if pool_size:
                kwargs['pool_size'] = pool_size
            if flip is not None:
                kwargs['flip'] = flip
            super().__init__(SpecGroup.SAMPLER, **kwargs)


class SimulationSpec(Specification):
    def __init__(self, n_iter: int, seeds: (int, list, np.ndarray), init_sample: np.ndarray,
                 thin: int = None, scaling: (list, np.ndarray) = None, reverse: bool = None):
        kwargs = dict(numiter=n_iter, seeds=np.array([seeds]).ravel(), init=init_sample)
        if thin:
            kwargs["thin"] = thin
        if scaling is not None:
            kwargs['scaling'] = scaling
        if reverse is not None:
            kwargs['reverse'] = reverse
        super().__init__(SpecGroup.SIMULATION, **kwargs)


class Data(Specification):
    def __init__(self, data: np.ndarray = None, seed: int = 0):
        if data is not None:
            super().__init__(SpecGroup.DATA, observations=data, dtype=str(data.dtype))
        else:
            super().__init__(SpecGroup.DATA, seed=seed)


class SmootherSpec(Specification):
    def __init__(self, filter_type: FilterType, smoother_type: SmootherType):
        super().__init__(SpecGroup.SMOOTHER, ftype=filter_type.value, stype=smoother_type.value)


class MCMCsession:
    def __init__(self, experiment_name: str):
        self._stem_ = experiment_name
        self._spec_file_ = experiment_name + "_specs.h5"
        self.samples = {}
        self.acceptances = {}
        self.data = None
        self.durations = {}

    def init(self, sequence_length: int, transition_model: TransitionSpec, observation_model: ObservationSpec,
             sampler: SamplerSpec, simulation_specs: SimulationSpec, data: Data, smoother: SmootherSpec = None):
        with h5py.File(DATA_PATH / self._spec_file_, 'w') as f:
            if transition_model.parametrised and observation_model.parametrised:
                f.attrs.create("parametrised", True, dtype=bool)
            model = f.create_group("model")
            model.attrs.create("length", sequence_length, dtype=int)
            transition_model.add2spec(model)
            observation_model.add2spec(model)
            sampler.add2spec(f)
            simulation_specs.add2spec(f)
            data.add2spec(f)
            if smoother is not None:
                smoother.add2spec(f)

    def run(self):
        rescode = None
        with sp.Popen(["../cmake-build-release/Baysis", self._spec_file_],
                      stdout=sp.PIPE, stderr=sp.STDOUT, bufsize=1, universal_newlines=True) as p:
            while True:
                line = p.stdout.readline()
                if not line:
                    break
                print(line, end="")
            rescode = p.poll()

        if rescode == 0:
            res_files = self.__get_resultsfiles__()
            self.__get_results__(res_files)
        else:
            return f"Programme terminated with error code {rescode}"

    def loadResults(self):
        res_files = self.__get_resultsfiles__()
        self.__get_results__(res_files)

    def hasResults(self):
        return len(self.__get_resultsfiles__()) > 0

    def __get_results__(self, res_files):
        for seed, rfile in res_files.items():
            try:
                with h5py.File(RESULTS_PATH / rfile, "r") as f:
                    print(f"Loading results for {seed}...", end="\t\t")
                    samples_ds = f['samples']
                    samples_shape = samples_ds.shape
                    self.samples[seed] = samples_ds[:].reshape(samples_shape[0], samples_shape[-1], -1)
                    self.durations[seed] = samples_ds.attrs['duration']
                    if self.data is None:
                        self.data = f['observations'][:].reshape(samples_shape[-1], -1)
                    self.acceptances[seed] = f['accepts'][:]
                    print("Done")
            except FileNotFoundError as fe:
                print(fe)
            except ValueError as e:
                print(e)

    def __get_resultsfiles__(self):
        fnames = {}
        for f in RESULTS_PATH.glob("*.h5"):
            if f.stem.startswith(self._stem_) and not f.stem.endswith("specs"):
                key = f.stem.split("_")[-1]
                fnames[key] = f.name
        return fnames

    # @staticmethod
    # def __get_filename__(model, sampler, simulation):
    #     fname = f'{str(sampler.stype)}'
    #     if sampler.stype == SamplerType.EHMM:
    #         fname += f'{sampler.pool_size}'
    #         try:
    #             flip = sampler.flip
    #             fname += "_flip" if flip else "_noflip"
    #         except AttributeError:
    #             fname += "_noflip"
    #     fname += f"_{str(model.mtype)}_"
    #     try:
    #         rev = simulation.reverse
    #         fname += "wreverse_" if rev else "noreverse_"
    #     except AttributeError:
    #         fname += "noreverse_"
    #     return fname


class SmootherResults:
    def __init__(self):
        try:
            with h5py.File(RESULTS_PATH / SMOOTHER_FNAME, "r") as f:
                means_ds = f['smoother/means']
                covs_ds = f['smoother/covariances']
                mshape = means_ds.shape
                cshape = covs_ds.shape
                dim = mshape[0]
                length = cshape[1] // dim
                self.means = means_ds[:].reshape(length, dim)
                self.covariances = np.vsplit(covs_ds[:].reshape(length, dim, -1), length)
        except FileNotFoundError as fe:
            print(fe)
        except ValueError as e:
            print(e)
