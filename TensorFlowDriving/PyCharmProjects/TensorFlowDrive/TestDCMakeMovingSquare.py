from libDonkeyCar.datastore import Tub
from libDonkeyCar.simulation import SquareBoxCamera, MovingSquareTelemetry


def create_sample_tub(path, records=10):
    inputs = ["cam/image_array", "user/angle", "user/throttle", "user/mode"]  # ['user/speed', 'cam/image']
    types = ["image_array", "float", "float", "str"]  # ['float', 'image']
    t = Tub(path, inputs=inputs, types=types)
    for _ in range(records):
        record = create_sample_record()
        t.put_record(record)
    return t


def create_sample_record():
    x, y = tel.run()
    img_arr = cam.run(x, y)
    mode = 'sim'
    return {'cam/image_array': img_arr, 'user/angle': (x-80)/80, 'user/throttle': y/120, "user/mode": mode}


path = 'C:\\Projects\\Robotics\\DonkeyCar\\DonkeySimWindows\\logsim'  # Sim or actual drive log location

cam = SquareBoxCamera(resolution=(120, 160), box_size=20, color=(255, 255, 255))
tel = MovingSquareTelemetry(max_velocity=29,
                            x_min=10, x_max=150,
                            y_min=10, y_max=110)

n_records=5000
print('...creating frames:', n_records)
create_sample_tub(path, records=n_records)
print('done, Tub at:', path)


