import csv
import pickle
import glob

filenames = glob.glob('model*.p')

items = []
for fname in filenames:
    p = pickle.load(open(fname, 'rb'))
    items.append({
        'filename': fname,
        'valid_score': p['best_score'],
        'C': p['classifier'].C,
        'color_space': p['extractor_parameters']['color_space'],
        'hist_feat': p['extractor_parameters']['hist_feat'],
        'spatial_feat': p['extractor_parameters']['spatial_feat'],
        'hist_bins': p['extractor_parameters']['hist_bins'],
        'hog_channel': p['extractor_parameters']['hog_channel'],
        'hog_orient': p['extractor_parameters']['hog_orient'],
        'hog_pix_per_cell': p['extractor_parameters']['hog_pix_per_cell'],
        'spatial_size': p['extractor_parameters']['spatial_size'][0],
        'accuracy': p['accuracy'],
        'features': p['features_shape'][1],
    })

with open('models.csv', 'w') as csvfile:
    fieldnames = items[0].keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(items)
