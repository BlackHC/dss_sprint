"""
Regression Datasets from https://www.sfu.ca/~ssurjano/index.html
"""
from dataclasses import dataclass

import numpy as np
import torch.utils.data
from sklearn.datasets import make_friedman1

from dss_sprint.utils.component import Interface


def example_f(x):
    return np.exp(0.5 * x - 0.5) + np.sin(1.5 * x)


class RegressionDataset(Interface):
    def get_XY(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def get_torch_dataset(self) -> torch.utils.data.Dataset:
        X, Y = self.get_XY()
        return torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32),
                                              torch.tensor(Y, dtype=torch.float32))


@dataclass
class ExampleF(RegressionDataset):
    n: int = 100

    def get_XY(self) -> tuple[np.ndarray, np.ndarray]:
        X = np.linspace(-12, 6, self.n)
        Y = example_f(X)
        return X, Y


@dataclass
class Higdon(RegressionDataset):
    """
    hig02 < - function(s)
    {
        ##########################################################################
        #
        # HIGDON (2002) FUNCTION
        #
        # Authors: Sonja Surjanovic, Simon Fraser University
        #          Derek Bingham, Simon Fraser University
        # Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
        #
        # Copyright 2013. Derek Bingham, Simon Fraser University.
        #
        # THERE IS NO WARRANTY, EXPRESS OR IMPLIED. WE DO NOT ASSUME ANY LIABILITY
        # FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
        # derivative works, such modified software should be clearly marked.
        # Additionally, this program is free software; you can redistribute it
        # and/or modify it under the terms of the GNU General Public License as
        # published by the Free Software Foundation; version 2.0 of the License.
        # Accordingly, this program is distributed in the hope that it will be
        # useful, but WITHOUT ANY WARRANTY; without even the implied warranty
        # of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
        # General Public License for more details.
        #
        # For function details and reference information, see:
        # http://www.sfu.ca/~ssurjano/
        #
        #########################################################################

        term1 < - sin(2 * pi * s / 10)
        term2 < - 0.2 * sin(2 * pi * s / 2.5)

        y < - term1 + term2
        return (y)
        }
    """
    n: int = 1000
    random_state: int = 0

    def get_XY(self) -> tuple[np.ndarray, np.ndarray]:
        X = np.linspace(0, 10, self.n)
        Y = np.sin(2 * np.pi * X / 10) + 0.2 * np.sin(2 * np.pi * X / 2.5)
        random = np.random.RandomState(self.random_state)
        return X, Y + random.normal(0, 0.1, size=self.n)


@dataclass
class Oako021d(RegressionDataset):
    """
    function [y] = oakoh021d(x)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % OAKLEY & O'HAGAN (2002) 1-DIMENSIONAL FUNCTION
    %
    % Authors: Sonja Surjanovic, Simon Fraser University
    %          Derek Bingham, Simon Fraser University
    % Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
    %
    % Copyright 2013. Derek Bingham, Simon Fraser University.
    %
    % THERE IS NO WARRANTY, EXPRESS OR IMPLIED. WE DO NOT ASSUME ANY LIABILITY
    % FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
    % derivative works, such modified software should be clearly marked.
    % Additionally, this program is free software; you can redistribute it
    % and/or modify it under the terms of the GNU General Public License as
    % published by the Free Software Foundation; version 2.0 of the License.
    % Accordingly, this program is distributed in the hope that it will be
    % useful, but WITHOUT ANY WARRANTY; without even the implied warranty
    % of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
    % General Public License for more details.
    %
    % For function details and reference information, see:
    % http://www.sfu.ca/~ssurjano/
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    y = 5 + x + cos(x);

    end
    """
    n: int = 100
    random_state: int = 0

    def get_XY(self) -> tuple[np.ndarray, np.ndarray]:
        random = np.random.RandomState(self.random_state)
        X = random.normal(0, 2, size=self.n)
        X = np.sort(X)
        Y = 5 + X + np.cos(X)

        return X, Y


@dataclass
class ForrEtAl08(RegressionDataset):
    """
    function [y] = forretal08(x)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % FORRESTER ET AL. (2008) FUNCTION
    %
    % Authors: Sonja Surjanovic, Simon Fraser University
    %          Derek Bingham, Simon Fraser University
    % Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
    %
    % Copyright 2013. Derek Bingham, Simon Fraser University.
    %
    % THERE IS NO WARRANTY, EXPRESS OR IMPLIED. WE DO NOT ASSUME ANY LIABILITY
    % FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
    % derivative works, such modified software should be clearly marked.
    % Additionally, this program is free software; you can redistribute it
    % and/or modify it under the terms of the GNU General Public License as
    % published by the Free Software Foundation; version 2.0 of the License.
    % Accordingly, this program is distributed in the hope that it will be
    % useful, but WITHOUT ANY WARRANTY; without even the implied warranty
    % of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
    % General Public License for more details.
    %
    % For function details and reference information, see:
    % http://www.sfu.ca/~ssurjano/
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    fact1 = (6*x - 2)^2;
    fact2 = sin(12*x - 4);

    y = fact1 * fact2;

    end
    """
    n: int = 100

    def get_XY(self) -> tuple[np.ndarray, np.ndarray]:
        X = np.linspace(0, 1, self.n)
        Y = (6 * X - 2)**2 * np.sin(12 * X - 4)
        return X, Y


@dataclass
class GramacyLee12(RegressionDataset):
    """
    function [y] = grlee12(x)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % GRAMACY & LEE (2012) FUNCTION
    %
    % Authors: Sonja Surjanovic, Simon Fraser University
    %          Derek Bingham, Simon Fraser University
    % Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
    %
    % Copyright 2013. Derek Bingham, Simon Fraser University.
    %
    % THERE IS NO WARRANTY, EXPRESS OR IMPLIED. WE DO NOT ASSUME ANY LIABILITY
    % FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
    % derivative works, such modified software should be clearly marked.
    % Additionally, this program is free software; you can redistribute it
    % and/or modify it under the terms of the GNU General Public License as
    % published by the Free Software Foundation; version 2.0 of the License.
    % Accordingly, this program is distributed in the hope that it will be
    % useful, but WITHOUT ANY WARRANTY; without even the implied warranty
    % of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
    % General Public License for more details.
    %
    % For function details and reference information, see:
    % http://www.sfu.ca/~ssurjano/
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    term1 = sin(10*pi*x) / (2*x);
    term2 = (x-1)^4;

    y = term1 + term2;

    end
    """
    n: int = 100

    def get_XY(self) -> tuple[np.ndarray, np.ndarray]:
        X = np.linspace(0.5, 2.5, self.n)
        Y = np.sin(10 * np.pi * X) / (2 * X) + (X - 1)**4
        return X, Y


@dataclass
class HolsEtAl13Sin(RegressionDataset):
    """
    function [y] = holsetal13sin(x)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % HOLSCLAW ET AL. (2013) SINUSOIDAL FUNCTION
    %
    % Authors: Sonja Surjanovic, Simon Fraser University
    %          Derek Bingham, Simon Fraser University
    % Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
    %
    % Copyright 2013. Derek Bingham, Simon Fraser University.
    %
    % THERE IS NO WARRANTY, EXPRESS OR IMPLIED. WE DO NOT ASSUME ANY LIABILITY
    % FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
    % derivative works, such modified software should be clearly marked.
    % Additionally, this program is free software; you can redistribute it
    % and/or modify it under the terms of the GNU General Public License as
    % published by the Free Software Foundation; version 2.0 of the License.
    % Accordingly, this program is distributed in the hope that it will be
    % useful, but WITHOUT ANY WARRANTY; without even the implied warranty
    % of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
    % General Public License for more details.
    %
    % For function details and reference information, see:
    % http://www.sfu.ca/~ssurjano/
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    y = x*sin(x) / 10;

    end
    """
    n: int = 1000
    random_state: int = 0

    def get_XY(self) -> tuple[np.ndarray, np.ndarray]:
        X = np.linspace(0, 10, self.n)
        Y = X * np.sin(X) / 10
        random = np.random.RandomState(self.random_state)
        Y += random.normal(0, 0.3, self.n)
        return X, Y


@dataclass
class SantnerEtAl03Dc(RegressionDataset):
    """
    function [y] = santetal03dc(x)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % SANTNER ET AL. (2003) DAMPED COSINE FUNCTION
    %
    % Authors: Sonja Surjanovic, Simon Fraser University
    %          Derek Bingham, Simon Fraser University
    % Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
    %
    % Copyright 2013. Derek Bingham, Simon Fraser University.
    %
    % THERE IS NO WARRANTY, EXPRESS OR IMPLIED. WE DO NOT ASSUME ANY LIABILITY
    % FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
    % derivative works, such modified software should be clearly marked.
    % Additionally, this program is free software; you can redistribute it
    % and/or modify it under the terms of the GNU General Public License as
    % published by the Free Software Foundation; version 2.0 of the License.
    % Accordingly, this program is distributed in the hope that it will be
    % useful, but WITHOUT ANY WARRANTY; without even the implied warranty
    % of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
    % General Public License for more details.
    %
    % For function details and reference information, see:
    % http://www.sfu.ca/~ssurjano/
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    fact1 = exp(-1.4*x);
    fact2 = cos(3.5*pi*x);

    y = fact1 * fact2;

    end
    """
    n: int = 100

    def get_XY(self) -> tuple[np.ndarray, np.ndarray]:
        X = np.linspace(0, 1, self.n)
        Y = np.exp(-1.4 * X) * np.cos(3.5 * np.pi * X)
        return X, Y



@dataclass
class CurrinEtAl88Sur(RegressionDataset):
    """
    function [y] = curretal88sur(x)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % CURRIN ET AL. (1988) SURVIVAL FUNCTION
    %
    % Authors: Sonja Surjanovic, Simon Fraser University
    %          Derek Bingham, Simon Fraser University
    % Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
    %
    % Copyright 2013. Derek Bingham, Simon Fraser University.
    %
    % THERE IS NO WARRANTY, EXPRESS OR IMPLIED. WE DO NOT ASSUME ANY LIABILITY
    % FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
    % derivative works, such modified software should be clearly marked.
    % Additionally, this program is free software; you can redistribute it
    % and/or modify it under the terms of the GNU General Public License as
    % published by the Free Software Foundation; version 2.0 of the License.
    % Accordingly, this program is distributed in the hope that it will be
    % useful, but WITHOUT ANY WARRANTY; without even the implied warranty
    % of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
    % General Public License for more details.
    %
    % For function details and reference information, see:
    % http://www.sfu.ca/~ssurjano/
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    y = 1 - exp(-1/(2*x));

    end
    """
    n: int = 100

    def get_XY(self) -> tuple[np.ndarray, np.ndarray]:
        X = np.linspace(0, 1, self.n)
        Y = 1 - np.exp(-1 / (2 * X))
        return X, Y


@dataclass
class Friedman1(RegressionDataset):
    n: int = 100
    n_features: int = 10
    noise: float = 0.0
    random_state: int = 0

    def get_XY(self) -> tuple[np.ndarray, np.ndarray]:
        return make_friedman1(n_samples=self.n, n_features=self.n_features, noise=self.noise, random_state=self.random_state)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset = ExampleF()
    X, Y = dataset.get_XY()
    plt.plot(X, Y)
    plt.title('Example F')
    plt.show()

    dataset = Higdon()
    X, Y = dataset.get_XY()
    plt.plot(X, Y)
    plt.title('Higdon')
    plt.show()

    dataset = Oako021d()
    X, Y = dataset.get_XY()
    plt.plot(X, Y)
    plt.title('Oako021d')
    plt.show()

    dataset = ForrEtAl08()
    X, Y = dataset.get_XY()
    plt.plot(X, Y)
    plt.title('Forretal08')
    plt.show()

    dataset = GramacyLee12()
    X, Y = dataset.get_XY()
    plt.plot(X, Y)
    plt.title('GramacyLee12')
    plt.show()

    dataset = HolsEtAl13Sin()
    X, Y = dataset.get_XY()
    plt.plot(X, Y)
    plt.title('HolsEtAl13Sin')
    plt.show()

    dataset = SantnerEtAl03Dc()
    X, Y = dataset.get_XY()
    plt.plot(X, Y)
    plt.title('SantnerEtAl03Dc')
    plt.show()

    dataset = CurrinEtAl88Sur()
    X, Y = dataset.get_XY()
    plt.plot(X, Y)
    plt.title('CurrinEtAl88Sur')
    plt.show()





