# import matplotlib.pyplot as plt
# import numpy as np
# import pickle

# with open('data.pkl', 'rb') as f:
#     loaded_data = pickle.load(f)

# da1 = loaded_data['IQhhADU'][10][10]
# # print(da)
# plt.plot(da1)
# plt.show()

# # Flatten
# arr = loaded_data['IQhhADU']
# arr_2d = arr.reshape(-1, arr.shape[2])
# arr_1d = arr_2d.flatten()
# # print("\n1D array:")
# # print(arr_1d.shape)

# # Convert back to 3d
# pulses = 83
# gates = 1000
# rays = 360

# # H component
# assert pulses * gates * rays == len(arr_1d)
# result_h = arr_1d.reshape(rays, gates, pulses)
# # print(result_h.shape)
# da = result_h[10][10]
# #print(da)
# plt.plot(da)
# plt.show()

import pickle
import matplotlib.pyplot as plt
import numpy as np 

with open('base@_test.pkl', 'rb') as f:
    loaded_data = pickle.load(f)


# Reflectivity Plot
theta = np.linspace(0, 2*np.pi, 360)
r = np.arange(150)

# Convert to Cartesian coordinates
R, Theta = np.meshgrid(r, theta)
X = R * np.cos(Theta)
Y = R * np.sin(Theta)

plt.figure(figsize=(16, 8))

# First subplot
plt.subplot(2, 2, 1)
plt.pcolormesh(X, Y, loaded_data['Reflectivity Horizontal'], cmap='viridis')
plt.colorbar(label='Reflectivity Horizontal(dBZ)')
plt.title('Radar Image 1')
plt.axis('equal')

# Second subplot
plt.subplot(2, 2, 2)
plt.pcolormesh(X, Y, loaded_data['Reflectivity Vertical'], cmap='viridis')
plt.colorbar(label='Reflectivity Vertical(dBZ)')
plt.title('Radar Image 2')
plt.axis('equal')

# Third subplot
plt.subplot(2, 2, 3)
plt.pcolormesh(X, Y, loaded_data['Differential Reflectivity'], cmap='viridis')
plt.colorbar(label='Differential Reflectivity (zDR)')
plt.title('Radar Image 3')
plt.axis('equal')

# fourth subplot
plt.subplot(2, 2, 4)
plt.pcolormesh(X, Y, loaded_data['RHO_HV'], cmap='viridis')
plt.colorbar(label='rho')
plt.title('Radar Image 4')
plt.axis('equal')

plt.show()



# plt.imshow(loaded_data, aspect='auto', origin='lower', cmap='viridis')
# plt.colorbar(label='Reflectivity (dBZ)')
# plt.title('Radar Image')
# plt.xlabel('Distance')
# plt.ylabel('Angle')
# plt.show()

