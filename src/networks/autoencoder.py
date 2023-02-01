import torch
import sparseconvnet as scn

from src.config.network import GrowthRate, DownSampling, UpSampling



def increase_filters(current_filters, params):
    if params.growth_rate == GrowthRate.multiplicative:
        return current_filters * 2
    else: # GrowthRate.additive
        return current_filters + params.n_initial_filters

def decrease_filters(current_filters, params):
    if params.growth_rate == GrowthRate.multiplicative:
        return current_filters // 2
    else: # GrowthRate.additive
        return current_filters - params.n_initial_filters




def create_models(params, image_size):


    if params.framework.sparse:
        from . sparse_building_blocks import Block, ResidualBlock
        from . sparse_building_blocks import ConvolutionDownsample, ConvolutionUpsample
        from . sparse_building_blocks import BlockSeries
    else:
        from . building_blocks import Block, ResidualBlock, BlockSeries
        from . building_blocks import ConvolutionDownsample, ConvolutionUpsample
        from . building_blocks import MaxPooling, InterpolationUpsample


    # Build the encoder:
    encoder = torch.nn.Sequential()

    if params.framework.sparse:
        input_layer = scn.InputLayer(dimension=3, spatial_size=torch.tensor(image_size))
    else:
        input_layer = torch.nn.Identity()


    encoder.append(
        Block(
            nIn    = 1,
            nOut   = params.encoder.n_initial_filters,
            params = params.encoder
        ),
    )

    # How many filters did we start with?
    current_number_of_filters = params.encoder.n_initial_filters

    if params.encoder.downsampling == DownSampling.convolutional:
        downsampler = ConvolutionDownsample
    else:
        downsampler = MaxPooling

    for i_layer in range(params.encoder.depth):
        next_filters = increase_filters(current_number_of_filters, params.encoder)
        layer = torch.nn.Sequential(
            downsampler(
                nIn    = current_number_of_filters,
                nOut   = next_filters,
                params = params.encoder
            ),
            BlockSeries(
                nIn      = next_filters,
                n_blocks = params.encoder.blocks_per_layer,
                params   = params.encoder
            )
        )
        current_number_of_filters = next_filters
        encoder.append(layer)

    # Now build the decoder:

    decoder = torch.nn.Sequential()


    if params.decoder.upsampling == UpSampling.convolutional:
        upsampler = ConvolutionUpsample
    else:
        upsampler = InterpolationUpsample

    for i_layer in range(params.decoder.depth):
        # Note that we pass the encoder params to decrease filters
        # Because we want to match the same rate of increase with a decrease
        next_filters = decrease_filters(current_number_of_filters, params.encoder)
        layer = torch.nn.Sequential(
            upsampler(
                nIn    = current_number_of_filters,
                nOut   = next_filters,
                params = params.decoder
            ),
            BlockSeries(
                nIn      = next_filters,
                n_blocks = params.decoder.blocks_per_layer,
                params   = params.decoder
            )
        )
        current_number_of_filters = next_filters
        decoder.append(layer)

    decoder.append(
        Block(
            nIn    = current_number_of_filters,
            nOut   = 1,
            params = params.decoder
        ),
    )

    # In sparse mode, spit out the final

    return input_layer, encoder, decoder
