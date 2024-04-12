
def show_max_validation_values(history_data):
    # バリデーションデータのキーのみを抽出
    validation_keys = [key for key in history_data.keys() if key.startswith('val_')]
    
    max_values = {}
    for key in validation_keys:
        # lossキーの場合は最小値を求める
        if 'loss' in key:
            min_value = min(history_data[key])
            max_values[key] = min_value
            print(f"Minimum {key}: {min_value}")
        else:
            # それ以外は最大値を求める
            max_value = max(history_data[key])
            max_values[key] = max_value
            print(f"Maximum {key}: {max_value}")

    return max_values
