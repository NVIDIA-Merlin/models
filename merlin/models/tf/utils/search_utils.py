def find_single_instance_in_layers(layer, type_to_search):
    if isinstance(layer, type_to_search):
        return layer

    if getattr(layer, "layers", None):
        for sub_layer in layer.layers:
            result = find_single_instance_in_layers(sub_layer, type_to_search)
            if result:
                return result

    return None


def find_all_instances_in_layers(layer, type_to_search):
    if isinstance(layer, type_to_search):
        return [layer]

    if getattr(layer, "layers", None):
        results = []
        for sub_layer in layer.layers:
            results.extend(find_all_instances_in_layers(sub_layer, type_to_search))

        return results

    return []
