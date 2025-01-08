# dataset.py
import numpy as np

def generate_xor_data(num_points=200, noise=0.5):
    """Generate points in range [-5,5], then label = 1 if x>0,y>0 or x<0,y<0."""
    xs = np.random.uniform(-5,5,size=(num_points,2))
    xs += np.random.normal(0, noise, size=xs.shape)
    labels = np.zeros(num_points, dtype=int)
    # label=1 if x>0,y>0 or x<0,y<0
    for i in range(num_points):
        x, y = xs[i]
        if (x>0 and y>0) or (x<0 and y<0):
            labels[i] = 1
    return xs, labels

def generate_spiral_data(num_points=200, noise=0.5):
    """Generate 2-arm spiral data."""
    # We'll split num_points in half: half in one arm, half in the other
    n = num_points // 2
    points = []
    labels = []
    def gen_spiral(delta_t, label):
        for i in range(n):
            r = float(i)/n*6.0
            t = 1.75*float(i)/n*2.0*np.pi + delta_t
            x = r*np.sin(t)
            y = r*np.cos(t)
            # add noise
            x += np.random.uniform(-1,1)*noise
            y += np.random.uniform(-1,1)*noise
            points.append([x,y])
            labels.append(label)

    # arm0 label=0, arm1 label=1
    gen_spiral(0, 0)
    gen_spiral(np.pi, 1)

    xs = np.array(points, dtype=np.float32)
    labs = np.array(labels, dtype=int)
    return xs, labs

def generate_gaussian_data(num_points=200, noise=0.5):
    """Generate two gaussians: one centered near (2,2) label=1, another near (-2,-2) label=0."""
    # half points in cluster1 => label=1, half in cluster0 => label=0
    n = num_points // 2
    xs = []
    labs= []
    # cluster1
    for i in range(n):
        x = np.random.normal(2, noise+1.0)
        y = np.random.normal(2, noise+1.0)
        xs.append([x,y])
        labs.append(1)
    # cluster0
    for i in range(n):
        x = np.random.normal(-2, noise+1.0)
        y = np.random.normal(-2, noise+1.0)
        xs.append([x,y])
        labs.append(0)
    xs = np.array(xs, dtype=np.float32)
    labs= np.array(labs, dtype=int)
    return xs, labs

def generate_circle_data(num_points=200, noise=0.5):
    """Generate points labeled 1 if inside radius*0.5, else 0 if outside radius."""
    radius = 5.0
    n = num_points//2
    points = []
    labels = []

    # inside circle => label=1
    for i in range(n):
        r = np.random.uniform(0, radius*0.5)
        angle = np.random.uniform(0, 2*np.pi)
        x = r*np.sin(angle)
        y = r*np.cos(angle)
        # add noise
        x += np.random.uniform(-radius,radius)*noise/3
        y += np.random.uniform(-radius,radius)*noise/3
        label = 1 if (x**2 + y**2 < (radius*0.5)**2) else 0
        points.append([x,y])
        labels.append(label)

    # outside circle => label=0
    for i in range(n):
        r = np.random.uniform(radius*0.75, radius)
        angle = np.random.uniform(0, 2*np.pi)
        x = r*np.sin(angle)
        y = r*np.cos(angle)
        x += np.random.uniform(-radius,radius)*noise/3
        y += np.random.uniform(-radius,radius)*noise/3
        label = 1 if (x**2 + y**2 < (radius*0.5)**2) else 0
        points.append([x,y])
        labels.append(label)

    xs = np.array(points, dtype=np.float32)
    labs= np.array(labels, dtype=int)
    return xs, labs


def generate_dataset(dataset_type="circle", num_points=200, noise=0.5):
    if dataset_type=="xor":
        return generate_xor_data(num_points, noise)
    elif dataset_type=="spiral":
        return generate_spiral_data(num_points, noise)
    elif dataset_type=="gaussian":
        return generate_gaussian_data(num_points, noise)
    else:
        # default circle
        return generate_circle_data(num_points, noise)


if __name__=="__main__":
    # Quick test
    xs, labs = generate_dataset("xor", 10, 0.1)
    print(xs, labs)
