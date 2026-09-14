"""
Physics-consistent data expansion for the FM-INR optical-fluence dataset.

Two routes, both producing physically consistent (optical_volume, illumination, Phi):
  - augment:  free rigid (flip/rot90) equivariant transforms of existing head scenes.
  - phantom + solvers: virtual geometric phantoms with a forward-solved fluence field.

See augment.py, solvers.py, phantom.py, generate.py.
"""
