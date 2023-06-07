def init_wiener_kernel(wiener_sz, wiener_init_path):
    wiener_kernel = io.loadmat(wiener_init_path)
    wiener_kernel = np.array(wiener_kernel['lin_hdr'])
    wiener_kernel = torch.from_numpy(wiener_kernel)
    wiener_kernel = wiener_kernel.permute(2, 0, 1)  # CHW
    '''
        crop to wiener_sz
    '''
    H = wiener_kernel.shape[1]
    W = wiener_kernel.shape[2]
    h, w = math.ceil(H/2), math.ceil(W/2)
    r = math.floor(wiener_sz/2)

    wiener_kernel = wiener_kernel[:, h-r:h+r, w-r:w+r]

    #print('psf_kernel',psf_kernel.shape)
    #assert False
    if wiener_kernel.shape[-1] % 2 == 0:  # kernel size must be odd
        wiener_kernel = F.pad(wiener_kernel, (0, 1, 0, 1), "constant", 0)
        # psf_kernel = psf_kernel[:, :-1, :-1]
    
    # rot 90
    wiener_kernel = wiener_kernel.permute(0, 2, 1) 
    
    # normalization
    # wiener_kernel = torch.div(wiener_kernel, torch.sum(wiener_kernel))

    return wiener_kernel.unsqueeze(0)
