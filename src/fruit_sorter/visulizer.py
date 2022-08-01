import matplotlib.pyplot as plt
import seaborn as sn

# create visulization functions for the data
def plot_data(data, labels, title):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.scatter(data[labels==0,0], data[labels==0,1], label='0', c='red')
    ax.scatter(data[labels==1,0], data[labels==1,1], label='1', c='blue')
    ax.set_title(title)
    ax.legend()
    # plt.show()
    plt.savefig(title + '.png')
    plt.close()
    return

def plot_feature_maps(model, data, labels, title):
    feature_maps = model.forward(data)
    num_feature_maps = feature_maps.shape[1]
    fig = plt.figure(figsize=(10,10))
    grid = plt.GridSpec(num_feature_maps, 1, hspace=0.5)
    for i in range(num_feature_maps):
        feature_map = feature_maps[:,i,:,:]
        ax = fig.add_subplot(grid[i,0])
        ax.imshow(feature_map.squeeze(), cmap='gray')
        ax.axis('off')
    plt.suptitle(title)
    # plt.show()
    plt.savefig(title + '.png')
    plt.close()
    return

