#
# cppbridge.py
# Baysis
#
# Created by Vladimir Sotskov on 28/08/2021, 12:02.
# Copyright © 2021 Vladimir Sotskov. All rights reserved.
#

import h5py
import subprocess as sp
import numpy as np
import pandas as pd
from enum import Enum, IntEnum, unique
from collections import defaultdict
from itertools import product
from pathlib import Path
from statsmodels.tools.decorators import cache_readonly

DATA_PATH = Path("../data")
RESULTS_PATH = Path("../data")
OUTPUTS_PATH = Path("../..").resolve() / "thesis/resources"


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
    def __new__(cls, value, label):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.label = label
        return obj

    NORMAL = (1, "Normal")
    UNIFORM = (2, "Uniform")
    INVGAMMA = (3, "Inverse Gamma")
    POISSON = (4, "Poisson")

    @property
    def nparams(self):
        if self == DistributionType.POISSON:
            return 1
        return 2

    def validate(self, params: tuple, support: tuple = None):
        if len(params) != self.nparams:
            raise ValueError(f"The {self.label} distribution must have 2 parameters; {len(params)} were given.")
        if np.any(np.array(params) <= 0) \
                and (self == DistributionType.POISSON or self == DistributionType.INVGAMMA):
            raise ValueError(f"The parameters for {self.label} distribution must be all positive.")
        if (support is None) and self != DistributionType.NORMAL:
            raise ValueError(f"The {self.label} distribution specification must include support.")
        if support is not None:
            if len(support) != 2:
                raise ValueError(f"Support should have two values, {len(support)} were given.")
            elif support[0] >= support[1]:
                raise ValueError(f"Incorrect support interval: {support[0]} > {support[1]}")

        if self == DistributionType.INVGAMMA:
            if params[0] <= 2:
                raise ValueError(f"The first parameter of the Inverse Gamma distribution (alpha) must be greater than 2"
                                 f" in order for vairance to be finite. Alpha supplied was {params[0]}.")
            print("Make sure the second parameter of Inverse Gamma distribution is the reciprocal of the scale.")

        if self == DistributionType.UNIFORM:
            if not np.all(np.array(params) == np.array(support)):
                raise ValueError(f"The parameters and support for {self.label} must coincide.")


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


def run_cpp_code(executable, specfile):
    rescode = None
    with sp.Popen([f"../bin/{executable}", specfile],
                  stdout=sp.PIPE, stderr=sp.STDOUT, bufsize=1, universal_newlines=True) as p:
        print(f"Launched {executable}")
        while True:
            line = p.stdout.readline()
            if not line:
                break
            print(line, end="")
        rescode = p.poll()

    return rescode


def hash_model(transitionm, observationm, seqlength):
    h = tuple()
    for m in [transitionm, observationm]:
        for attr, value in vars(m).items():
            if value is not None and attr[0] != "_":
                if isinstance(value, list):
                    h += tuple(value)
                elif isinstance(value, np.ndarray):
                    h += tuple(value.flatten())
                else:
                    h += tuple([value])
    h += tuple([seqlength])
    return f"{ModelType(transitionm.mtype)}_{ModelType(observationm.mtype)}_{abs(hash(h))}"


###
# Main classes
###
class Specification:
    def __init__(self, specname: Enum, **kwargs):
        self._groupname_ = specname.value
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    def add2spec(self, spec):
        grp = spec.create_group(self._groupname_)
        for attr, value in vars(self).items():
            if value is not None and attr[0] != "_":
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

    def parametrize(self, dim: int, distribution: DistributionType, prior: tuple, *args, varscale=None, minx=None,
                    maxx=None):
        support = None if (minx is None) or (maxx is None) else (minx, maxx)
        distribution.validate(prior, support)
        if self._type_ == ParamType.SYMMETRIC_MATRIX and len(args) == 0:
            raise AttributeError("Symmetric Matrix parameters must include the value for the main diagonal. "
                                 "This value is fixed and not sampled during the simulation.")
        elif self._type_ != ParamType.SYMMETRIC_MATRIX and len(args) > 0:
            raise AttributeError(f"Extra arguments for {self._type_} not supported.")

        self.parametrized = True
        self.dim = dim
        param_id = self._type_ * 10 + distribution.value
        eps = 1. if varscale is None else varscale
        self._value_ = [param_id, 0 if minx is None else minx, 0 if maxx is None else maxx] + [eps] \
                       + list(prior) + list(args)

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
        self.dim = len(val)

    @property
    def id(self):
        return self._value_[0]


class SymmetricMatrixParam(ModelParameter):
    def __init__(self):
        super().__init__(ParamType.SYMMETRIC_MATRIX)


class DiagonalMatrixParam(ModelParameter):
    def __init__(self):
        super().__init__(ParamType.DIAGONAL_MATRIX)


class VectorParam(ModelParameter):
    def __init__(self):
        super().__init__(ParamType.VECTOR)

    @property
    def value(self):
        if self.parametrized:
            return np.array(self._value_)
        else:
            return self._value_.reshape(-1, 1)

    @value.setter
    def value(self, val):
        super(VectorParam, type(self)).value.fset(self, val)


class ConstParam(ModelParameter):
    def __init__(self, value: np.ndarray, parametrised=False):
        ptype = ParamType.CONST_VECTOR if value.ndim == 1 \
            else ParamType.CONST_MATRIX if value.ndim == 2 \
            else ValueError(f"Only vector (1-d array) or matrix (2-d array) are allowed. "
                            f"Array with {value.ndim} dimensions was provided.")
        super().__init__(ptype)
        self.parametrized = parametrised
        self.dim = value.shape[-1]
        self._value_ = value

    @property
    def value(self):
        if self.parametrized:
            return [self._type_, self._value_.flatten()[0]]
        elif self._type_ == ParamType.CONST_VECTOR:
            return self._value_.reshape(-1, 1)
        else:
            return super().value()

    @property
    def id(self):
        return self._type_ * 10
    

class TransitionSpec(Specification):
    def __init__(self, mean_coeff: ModelParameter, cov: ModelParameter, prior_mean: np.ndarray, prior_cov: np.ndarray):
        self.parametrised = mean_coeff.parametrized | cov.parametrized
        mtype = ModelType.LINEAR_GAUSS.value
        if self.parametrised:
            for i, p in enumerate([mean_coeff, cov]):
                mtype += pow(10, 2 * i + 1) * p.id
        super().__init__(SpecGroup.TRANSITION_MODEL, mtype=mtype, A=mean_coeff.value, Q=cov.value,
                         xdim=mean_coeff.dim, mu_prior=prior_mean.reshape(-1, 1), S_prior=prior_cov)


class ObservationSpec(Specification):
    def __init__(self, model_type: ModelType, mean_coeff: ModelParameter, *params: ModelParameter):
        pp = [mean_coeff] + list(params)
        self.parametrised = any([p.parametrized for p in pp])
        mtype = model_type.value
        if self.parametrised:
            for i, p in enumerate(pp):
                mtype += pow(10, 2 * i + 1) * p.id
        if model_type == ModelType.LINEAR_GAUSS:
            super().__init__(SpecGroup.OBSERVATION_MODEL, mtype=mtype,
                             C=mean_coeff.value, R=params[0].value, ydim=mean_coeff.dim)
        elif model_type == ModelType.LINEAR_POISSON:
            super().__init__(SpecGroup.OBSERVATION_MODEL, mtype=mtype,
                             C=mean_coeff.value, D=params[0].value, controls=params[1].value,
                             ydim=mean_coeff.dim, cdim=params[1].dim)
        elif model_type == ModelType.BIMODAL_POISSON:
            super().__init__(SpecGroup.OBSERVATION_MODEL, mtype=mtype,
                             C=mean_coeff.value, mean_function=model_type.mean_function(), ydim=mean_coeff.dim)


class SamplerSpec(Specification):
    def __init__(self, sampler_type: SamplerType, *,
                 num_parameter_updates: int = None, pool_size: int = None, flip: bool = None):
        if sampler_type == SamplerType.METROPOLIS:
            super().__init__(SpecGroup.SAMPLER, stype=sampler_type, num_param_updates=num_parameter_updates)
        else:
            kwargs = dict(stype=sampler_type, num_param_updates=num_parameter_updates)
            if pool_size is not None:
                kwargs['pool_size'] = pool_size
            if flip is not None:
                kwargs['flip'] = flip
            super().__init__(SpecGroup.SAMPLER, **kwargs)


class SimulationSpec(Specification):
    def __init__(self, n_iter: int, seeds: (int, list, np.ndarray), init_sample: np.ndarray, *,
                 thinning: int = None, scaling: (list, np.ndarray) = None, reverse: bool = None):
        kwargs = dict(numiter=n_iter, seeds=np.array([seeds]).ravel(), init=init_sample)
        if thinning:
            kwargs["thin"] = thinning
        if scaling is not None:
            kwargs['scaling'] = scaling
        if reverse is not None:
            kwargs['reverse'] = reverse
        super().__init__(SpecGroup.SIMULATION, **kwargs)


class DataGenerator:
    def __init__(self, tm: TransitionSpec, om: ObservationSpec, seqlngth, seed: int):
        if tm.parametrised or om.parametrised:
            raise ValueError("Data cannot be generated from the models with unknown parameters.")

        self.modelhash = hash_model(tm, om, seqlngth)
        self.specfile = DATA_PATH / (self.modelhash + "_data.h5")
        self._states_ = None
        self._observations_ = None
        self.generated = False

        if not self.specfile.exists():
            with h5py.File(self.specfile, 'w') as f:
                model = f.create_group("model")
                model.attrs.create("length", seqlngth, dtype=int)
                tm.add2spec(model)
                om.add2spec(model)
                data = f.create_group(SpecGroup.DATA.value)
                data.attrs.create("seed", seed, dtype=int)
        else:
            with h5py.File(self.specfile, 'r') as f:
                if 'observations' in f.keys():
                    self.generated = True

    def generate(self):
        if self.generated:
            print("The data for these models have been generated already.")
            self.generated = True
            return

        rescode = run_cpp_code("datagen", self.specfile)

        if rescode != 0:
            return f"Programme terminated with error code {rescode}"
        else:
            self.generated = True

    @property
    def states(self):
        if self.generated:
            if self._states_ is not None:
                return self._states_
            else:
                with h5py.File(self.specfile, 'r') as f:
                    self._states_ = f['states'][:]
                    return self._states_
        else:
            raise ValueError("No data is generated for this model yet. Run generate() first.")

    @property
    def observations(self):
        if self.generated:
            if self._observations_ is not None:
                return self._observations_
            else:
                with h5py.File(self.specfile, 'r') as f:
                    self._observations_ = f['observations'][:]
                    return self._observations_
        else:
            raise ValueError("No data is generated for this model yet. Run generate() first.")


class Data(Specification):
    def __init__(self, data_provider: (str, Path)):
        self._datafile = DATA_PATH / data_provider
        if not self._datafile.exists():
            raise ValueError("The data for this model have not been generated. "
                             "Please use DataGenerator to generate the data first.")
        with h5py.File(DATA_PATH / data_provider, 'r') as f:
            dtype = str(f['observations'].dtype)
        super().__init__(SpecGroup.DATA, dref=data_provider, dtype=dtype)
        self._modelhash = "_".join(str(data_provider).split("_")[:-1])

    def inspect(self):
        with h5py.File(self._datafile, 'r') as f:
            trm = f['model/transition']
            obm = f['model/observation']
            trmstr = f"Transition model, size {trm.attrs['xdim']}:"
            obmstr = str(ModelType(obm.attrs['mtype'])).capitalize() + f" observation model, size {obm.attrs['ydim']}"
            print("".join(['#'] * 50))
            print(trmstr)
            for attr in trm.keys():
                print(attr)
                print(trm[attr][:])
            print("".join(['#'] * 50))
            print(obmstr)
            for attr in obm.keys():
                print(attr)
                print(obm[attr][:])


class SmootherSpec(Specification):
    def __init__(self, filter_type: FilterType, smoother_type: SmootherType):
        super().__init__(SpecGroup.SMOOTHER, ftype=filter_type.value, stype=smoother_type.value)


class KalmanSession:
    def __init__(self, specname: str):
        self._stem_ = specname
        self._spec_file_ = specname + "_specs.h5"
        self.means = None
        self.covariances = None

    def init(self, seql: int, transition_model: TransitionSpec, observation_model: ObservationSpec,
             smoother: SmootherSpec, data: Data):
        mhash = hash_model(transition_model, observation_model, seql)
        if mhash != data._modelhash:
            raise ValueError("The specification of the models is different form the model that generated the data.")

        if transition_model.parametrised or observation_model.parametrised:
            raise NotImplementedError("Only models with known parameters are implemented for Kalman smoothing.")

        with h5py.File(DATA_PATH / self._spec_file_, 'w') as f:
            model = f.create_group("model")
            model.attrs.create("length", seql, dtype=int)
            transition_model.add2spec(model)
            observation_model.add2spec(model)
            smoother.add2spec(f)
            data.add2spec(f)

    def run(self):
        rescode = run_cpp_code("Kalman", self._spec_file_)

        if rescode == 0:
            self.__get_results__()
        else:
            return f"Programme terminated with error code {rescode}"

    def hasResults(self):
        return (RESULTS_PATH / self._spec_file_).exists()

    def loadResults(self):
        self.__get_results__()

    def __get_results__(self):
        res_file = self._stem_ + "_results.h5"
        try:
            with h5py.File(RESULTS_PATH / res_file, "r") as f:
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


class MCMCsession:
    def __init__(self, experiment_name: str):
        self._stem_ = experiment_name
        self._spec_file_ = experiment_name + "_specs.h5"
        self.samples = {}
        self.acceptances = {}
        self.durations = {}
        self.param_samples = {}
        self.param_acceptances = {}
        self.nparams = 0
        if Path(DATA_PATH / self._spec_file_).exists():
            # We're retrieving results of previously run experiment
            with h5py.File(DATA_PATH / self._spec_file_, 'r') as f:
                self.__set_nparams__(f)

    def init(self, sequence_length: int, transition_model: TransitionSpec, observation_model: ObservationSpec,
             sampler: SamplerSpec, simulation_specs: SimulationSpec, data: Data):
        if not (transition_model.parametrised or observation_model.parametrised):
            mhash = hash_model(transition_model, observation_model, sequence_length)
            if mhash != data._modelhash:
                raise ValueError("The specification of the models is different from the model that generated the data.")

        with h5py.File(DATA_PATH / self._spec_file_, 'w') as f:
            f.attrs.create("mcmc_id",
                           sampler.stype + 100 * transition_model.mtype + 10 * observation_model.mtype,
                           dtype=int)
            if transition_model.parametrised and observation_model.parametrised:
                f.attrs.create("parametrised", True, dtype=bool)
                self.nparams = observation_model
                if sampler.num_param_updates is None:
                    raise AttributeError(
                        "The sampler for parametrized models has to have 'num_param_updates' attribute")
            model = f.create_group("model")
            model.attrs.create("length", sequence_length, dtype=int)
            transition_model.add2spec(model)
            observation_model.add2spec(model)
            sampler.add2spec(f)
            simulation_specs.add2spec(f)
            data.add2spec(f)
            self.__set_nparams__(f)

    def run(self):
        rescode = run_cpp_code("Baysis", self._spec_file_)

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

    def getSamples(self, burnin):
        return np.concatenate([s[burnin:] for _, s in self.samples.items()])

    def getParamSamples(self, burnin, param_names, forseed=None):
        if forseed is None:
            return pd.DataFrame(np.concatenate([s[burnin:] for _, s in self.param_samples.items()]),
                                columns=param_names)
        else:
            try:
                return pd.DataFrame(self.param_samples[f"seed{forseed}"][burnin:], columns=param_names)
            except KeyError:
                raise KeyError(f"No samples for this seed {forseed}")

    def __get_results__(self, res_files):
        for seed, rfile in res_files.items():
            try:
                with h5py.File(RESULTS_PATH / rfile, "r") as f:
                    print(f"Loading results for {seed}...", end="\t\t")
                    samples_ds = f['samples']
                    samples_shape = samples_ds.shape
                    self.samples[seed] = samples_ds[:].reshape(samples_shape[0], samples_shape[-1], -1)
                    self.durations[seed] = samples_ds.attrs['duration']
                    self.acceptances[seed] = f['accepts'][:]
                    parsamples_ds = f.get('par_samples')
                    if parsamples_ds and ('trm_par_acceptances' in parsamples_ds.attrs) \
                            and ('obsm_par_acceptances' in parsamples_ds.attrs):
                        self.param_samples[seed] = parsamples_ds[:, :self.nparams].squeeze(axis=2)
                        parmacc = dict(trm=parsamples_ds.attrs['trm_par_acceptances'],
                                       obsm=parsamples_ds.attrs['obsm_par_acceptances'])
                        self.param_acceptances[seed] = parmacc
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

    def __set_nparams__(self, h5f):
        if h5f.attrs.get('parametrised', False):
            self.nparams = 2
            for _, v in h5f['model/observation'].items():
                if v[0] != ParamType.CONST_VECTOR and v[0] != ParamType.CONST_MATRIX:
                    self.nparams += 1


###
# Helper class to check model and session ids
###
class Singleton(type):
    """ Use as metaclass in class signature: metaclass=Singleton"""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class IdChecker(metaclass=Singleton):
    models_map = {"trm": [ParamType.DIAGONAL_MATRIX, ParamType.SYMMETRIC_MATRIX],
                  ModelType.LINEAR_GAUSS: [ParamType.DIAGONAL_MATRIX, ParamType.DIAGONAL_MATRIX],
                  ModelType.LINEAR_POISSON: [ParamType.DIAGONAL_MATRIX, ParamType.DIAGONAL_MATRIX,
                                             ParamType.CONST_VECTOR],
                  ModelType.BIMODAL_POISSON: [ParamType.VECTOR]}

    def __init__(self):
        self._cache = {}

    @cache_readonly
    def params(self):
        res = {}
        for p, d in product(ParamType, DistributionType):
            _id = p * 10
            n = f"{p.name}"
            if p != ParamType.CONST_MATRIX and p != ParamType.CONST_VECTOR:
                _id += d
                n += f"<{d.name}>"
            res.setdefault(p, set()).add((n, _id))
        return res

    @cache_readonly
    def models(self):
        prms = self.params
        trms = []
        obms = []

        for model, plst in IdChecker.models_map.items():
            if model == "trm":
                mid = 1
                mname = ModelType.LINEAR_GAUSS.name
            else:
                mid = model
                mname = model.name
            for combo in product(*[prms[k] for k in plst]):
                _id = sum([x[1] * pow(10, i * 2 + 1) for i, x in enumerate(combo)]) + mid
                n = mname + "<" + ",".join([x[0] for x in combo]) + " >"
                if model == "trm":
                    trms.append((n, _id))
                else:
                    obms.append((n, _id))
        return trms, obms

    @cache_readonly
    def samplers(self):
        tm, om = self.models
        res = dict()
        for s in SamplerType:
            for mcombo in product(tm, om):
                names, ids = zip(*mcombo)
                res.setdefault("Name", []).append(s.name + "<" + ",".join(names) + " >")
                res.setdefault("Id", []).append(ids[0] * 100 + ids[1] * 10 + s)
        res = pd.DataFrame(res).set_index("Id")
        return res

    def checkSpecs(self, mcmc_specs: MCMCsession, verbose=False):
        with h5py.File(DATA_PATH / mcmc_specs._spec_file_, 'r') as f:
            if not f.attrs['parametrised']:
                print("Check is enabled for parametrised models only")
                return
            mcmc_id = f.attrs['mcmc_id']
            tmodel_id = f['model/transition'].attrs['mtype']
            omodel_id = f['model/observation'].attrs['mtype']

        tm, om = self.models
        tmodel_correct = pd.DataFrame({"dummy": 0}, index=pd.MultiIndex.from_tuples(tm)) \
            .reset_index(level=0).drop("dummy", axis=1).to_dict()['level_0'].get(tmodel_id, False)
        omodel_correct = pd.DataFrame({"dummy": 0}, index=pd.MultiIndex.from_tuples(om)) \
            .reset_index(level=0).drop("dummy", axis=1).to_dict()['level_0'].get(omodel_id, False)
        mcmc_correct = self.samplers.to_dict()["Name"].get(mcmc_id, False)

        msg = f"""
        Transition model is {tmodel_id}: {tmodel_correct}
        Observation model is {omodel_id}: {omodel_correct}
        MCMC sampler is {mcmc_id}: {mcmc_correct}"""
        if not (tmodel_correct and omodel_correct and mcmc_correct):
            raise ValueError(f"Incorrect specification!{msg}")
        elif verbose:
            print(msg)
        return True
