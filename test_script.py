from sefa.models import parse_gan_type
from sefa.utils import load_generator, get_weights, factorize_weight, imshow, sample, synthesize

model_name = "stylegan_animeface512" # ['stylegan_animeface512', 'stylegan_car512', 'stylegan_cat256', 'pggan_celebahq1024', 'stylegan_bedroom256']

generator = load_generator(model_name)
gan_type = parse_gan_type(generator)
num_samples = 3 # 1-8
noise_seed = 0 # 0-1000
layer_idx = "all" # ['all', '0-1', '2-5', '6-13']
weights = get_weights(generator, layer_idx)


# SeFa algorithm - Factorization

# codes = sample(generator, gan_type, num_samples, noise_seed)
# images = synthesize(generator, gan_type, codes)
# imshow(images, col=num_samples)

# layers, boundaries, _ = factorize_weight(generator, layer_idx) # boundaries = eigenvectors

# new_codes = codes.copy()
# for sem_idx in range(5):
#   boundary = boundaries[sem_idx:sem_idx + 1]
#   step = eval(f'semantic_{sem_idx + 1}')
#   if gan_type in ['stylegan', 'stylegan2']:
#     new_codes[:, layers, :] += boundary * step
# new_images = synthesize(generator, gan_type, new_codes)
# imshow(new_images, col=num_samples)