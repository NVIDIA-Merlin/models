def find_single_instance_in_layers(layer, to_search):
    if isinstance(layer, to_search):
        return layer
    elif layer == to_search:
        return layer

    if getattr(layer, "layers", None):
        for sub_layer in layer.layers:
            result = find_single_instance_in_layers(sub_layer, to_search)
            if result:
                return result

    return None


def find_all_instances_in_layers(layer, to_search):
    if isinstance(layer, to_search):
        return [layer]
    elif layer == to_search:
        return [layer]

    if getattr(layer, "layers", None):
        results = []
        for sub_layer in layer.layers:
            results.extend(find_all_instances_in_layers(sub_layer, to_search))

        return results

    return []


def replace_all_instances_in_layers(layer, to_search, to_replace):
    if layer == to_search:
        return True

    if getattr(layer, "layers", None):
        layers_to_replace = []
        for i, sub_layer in enumerate(layer.layers):
            need_to_replace = replace_all_instances_in_layers(sub_layer, to_search, to_replace)
            if need_to_replace:
                layers_to_replace.append(i)

        for i in layers_to_replace:
            layer.layers[i] = to_replace

    return False
