import odl
import numpy as np


class initialization:
    def __init__(self, spacing):
        self.param = {}
        self.reso = 512 / 256 * spacing

        # image
        self.param['nx_h'] = 256
        self.param['ny_h'] = 256
        self.param['sx'] = self.param['nx_h']*self.reso
        self.param['sy'] = self.param['ny_h']*self.reso

        ## view
        self.param['startangle'] = 0
        self.param['endangle'] = 2 * np.pi

        self.param['nProj'] = 320

        ## detector
        self.param['su'] = 2*np.sqrt(self.param['sx']**2+self.param['sy']**2)
        self.param['nu_h'] = 321
        self.param['dde'] = 1075*self.reso
        self.param['dso'] = 1075*self.reso

        self.param['u_water'] = 0.192


def build_gemotry(param):
    reco_space_h = odl.uniform_discr(
        min_pt=[-param.param['sx'] / 2.0, -param.param['sy'] / 2.0],
        max_pt=[param.param['sx'] / 2.0, param.param['sy'] / 2.0], shape=[param.param['nx_h'], param.param['ny_h']],
        dtype='float32')

    angle_partition = odl.uniform_partition(param.param['startangle'], param.param['endangle'],
                                            param.param['nProj'])

    detector_partition_h = odl.uniform_partition(-(param.param['su'] / 2.0), (param.param['su'] / 2.0),
                                                 param.param['nu_h'])

    geometry_h = odl.tomo.FanBeamGeometry(angle_partition, detector_partition_h,
                                          src_radius=param.param['dso'],
                                          det_radius=param.param['dde'])

    ray_trafo_hh = odl.tomo.RayTransform(reco_space_h, geometry_h, impl='astra_cuda')

    # Fourier transform in detector direction
    fourier = odl.trafos.FourierTransform(ray_trafo_hh.range, axes=[1])
    # Create ramp in the detector direction
    ramp_function = fourier.range.element(lambda x: np.abs(x[1]) / (2 * np.pi))
    # Create ramp filter via the convolution formula with fourier transforms
    ramp_filter = fourier.inverse * ramp_function * fourier
    # Create filtered back-projection by composing the back-projection (adjoint)
    # with the ramp filter.
    fbp = ray_trafo_hh.adjoint * ramp_filter

    return ray_trafo_hh, fbp
