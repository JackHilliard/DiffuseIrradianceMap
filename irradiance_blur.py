"""
    An informative explanation of the reflectance equation for IBL and PBR can be found here:
        https://learnopengl.com/PBR/Theory
    Rotations derived from:
        https://github.com/BlueHorn07/pyEquirectRotate
    Spherical Grids are from:
        https://github.com/VCL3D/SphericalViewSynthesis
"""

import math
import spherical as SPH
import torch

"""
Coordinate system:
    cartesian on equirectangular pano:
    -   ranges from 0 to h/w
    -   starts at bottom left goes to top right

    lat lon on equirectangular pano:
    -   lat ranges from -pi to pi, starts at h to 0
    -   lon ranges from -pi to pi, starts at 0 goes to w

    polar coordinates:
    -   elevation starts at h, goes to 0
    -   azimuth starts at 0, goes to w
"""
class irradiance_blur():
    def __init__(self, dims, device):
        self.batch_size = dims[0]
        self.colour = dims[1]
        self.height = dims[2]
        self.width  = dims[3]
        self.device = device
        self.half_height = int(math.floor(self.height/2))

        #generate attention weights and cosine weights
        spherical_weights = SPH.weights.spherical_confidence(SPH.grid.create_spherical_grid(self.width)).to(self.device).squeeze(0)
        self.attention_weights = spherical_weights[0][:self.half_height][:] #1,H,W
        phi_weights =  SPH.weights.theta_confidence(SPH.grid.create_spherical_grid(self.width)).to(self.device).squeeze(0)
        phi_hemisphere = phi_weights[0][:self.half_height][:]
        self.cosine_weights = torch.ones_like(phi_hemisphere) - phi_hemisphere

        # mapping equirect coordinate into LatLon coordinate system
        out_LonLat = torch_Pixel2LatLon(self.height,self.width,self.device)  # (H, W, (lat, lon))
        # mapping LatLon coordinate into xyz(sphere) coordinate system
        self.out_xyz = torch_LatLon2Sphere(out_LonLat)  # (H, W, (x, y, z))
        #convert lat lon to rotation matrix format
        out_LonLat[:,:,1] = out_LonLat[:,:,1] #longitude
        out_LonLat[:,:,0] = -1*(out_LonLat[:,:,0] - (math.pi/2))    #latitude
        #calculate all rotation matrices
        #rotation is gathered by using the negative of the lat lon coords
        self.Rt = torch_getAllRotMatrix(out_LonLat,self.device)


    def __call__(self,image, alpha, e):
        #image: (B, C, H, W)
        #output: (B, H, W, C)
        output = torch.zeros_like(image).permute((2,3,0,1)) #(H, W, B, C)
        hdr = image.flatten(2)
        #setup cosine power weights
        powCos = torch.pow(self.cosine_weights,alpha*e)
        #get index of first value lower than the minimum value cosine weights
        for y in range(powCos.shape[0]):
            if powCos[y][0].item() <= self.cosine_weights.min().item():
                 cutoff = y
                 break
        powCos_weights = powCos[...,:cutoff,:]
        for y in range(self.height):
            for x in range(self.width):
                #rotate spherical coordinates
                src_Pixel = torch_LatLon2Pixel(torch_Sphere2LatLon(torch.matmul(self.out_xyz, self.Rt[y][x])))  # (H, W, 2)
                z = src_Pixel[..., 0]*self.width + src_Pixel[..., 1]

                hemisphere_new = torch.gather(hdr, 2, z.flatten(0).repeat(self.batch_size, self.colour,1)).reshape_as(image)    # (B, C, H, W)
                hemisphere = hemisphere_new[...,:cutoff,:]

                output[y][x] = torch.mean(hemisphere * powCos_weights, (2,3)) # (B, C)

        return output.permute((2,3,0,1))
        


def torch_getAllRotMatrix(rotation, device):#rotation
    """
    :param rotation: (yaw, pitch, roll) in degree
    :return: general rotational matrix
    refer this: https://en.wikipedia.org/wiki/Rotation_matrix#General_rotations
    """
    #input rotation shape:  [y, x, 2]
    #output shape: [y, x, 3, 3]
    Lat = rotation[:, :, 0].flatten()
    Lon = rotation[:, :, 1].flatten()

    zeros = torch.zeros_like(Lat)
    Rzyt = torch.stack([torch.stack([torch.cos(Lon)*torch.cos(Lat), torch.sin(Lon)*torch.cos(Lat), -torch.sin(Lat)]),
                        torch.stack([-torch.sin(Lon), torch.cos(Lon), zeros]),
                        torch.stack([torch.cos(Lon)*torch.sin(Lat), torch.sin(Lon)*torch.sin(Lat), torch.cos(Lat)])]).permute(2,0,1)

    output = Rzyt.reshape((rotation.shape[0],rotation.shape[1],3,3))
    return output

def torch_Pixel2LatLon(h, w, device):
  # LatLon (H, W, (lat, lon))
  Lat = (0.5 - torch.arange(0, h,dtype=torch.float,device=device)/h) * math.pi
  Lon = (torch.arange(0, w,dtype=torch.float,device=device)/w - 0.5) * 2 * math.pi

  Lat = torch.tile(Lat.unsqueeze(1), (w,))
  Lon = torch.tile(Lon, (h, 1))

  return torch.dstack((Lat, Lon))

def torch_LatLon2Sphere(LatLon):
  Lat = LatLon[:, :, 0]
  Lon = LatLon[:, :, 1]
  x = torch.cos(Lat) * torch.cos(Lon)
  y = torch.cos(Lat) * torch.sin(Lon)
  z = torch.sin(Lat)

  return torch.dstack((x, y, z))

def torch_Sphere2LatLon(xyz):
  Lat = math.pi / 2 - torch.arccos(xyz[:, :, 2])
  Lon = torch.atan2(xyz[:, :, 1], xyz[:, :, 0])

  return torch.dstack((Lat, Lon))

def torch_LatLon2Pixel(LatLon):
  h, w, _ = LatLon.shape
  Lat = LatLon[:, :, 0]
  Lon = LatLon[:, :, 1]
  i = (h * (0.5 - Lat / math.pi)) % h
  j = (w * (0.5 + Lon / (2 * math.pi))) % w

  return torch.dstack((i, j)).type(torch.int64)
