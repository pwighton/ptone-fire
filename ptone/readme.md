## Configuration
 
### On the scanner

- Connect the python-ismrmrd-server computer so that it can communicate with mars (192.168.2.2)
- On the scanner, make a directory under `%CustomerIceProgs%`, (e.g. `pwighton`)
- Copy the following files from `ptone/scanner-config` to this directory
  - `wip_070_fire_IceFireRawAddin_PilotTone.ipr`
  - `wip_070_fire_IceFireRawAddin_PilotTone.xml`
  - `wip_070_fire_PilotTone.ini`
- Edit the files
  - `wip_070_fire_PilotTone.ini`:
    - Change `hostname` to the IP address of the python-ismrmrd-server
    - Change `port` to the port of the python-ismrmrd-server
    - Nothing under `[chroot]` matters unless you are running the python-ismrmrd-server on mars via gadgetron
  - `wip_070_fire_IceFireRawAddin_PilotTone.xml`:
    - Change `<IniFile>` to the location of the `wip_070_fire_PilotTone.ini` file
    - No Other edits required
  - `wip_070_fire_IceFireRawAddin_PilotTone.ipr`:
    - no edits required

### On the python-ismrmrd-server computer

Set up the conda environment for the python-ismrmrd-server
```
conda env create -f environment.yml
```

Activate the environment
```
conda activate mrd
```

Set up the conda environment for USRP transmission.  Due to a boost version conflict
with python-ismrmrd-server, this is in a seperate environment
```
conda env create -f ptone/environment-tx.yml
```

Test the environment
```
conda activate ptone-tx
uhd_images_downloader -t b2xx   # firmware images; kstream's environment has exactly the B2xx set
uhd_find_devices                # check the USRP is found
```

Edit `pilottone.json`
- These are the settings for live scans. Where the server gets them from:
  - If the scanner sends a JSON config with the scan (it does if it has `pilottone.json` in its
    `fire\config` folder, or one named in `<JsonConfig>`), that is used.
  - If it sends none, the server uses its own `pilottone.json` (the one in this repository), or the file
    named by the environment variable `PTONE_FALLBACK_CONFIG`.
  - Offline, `mrd-client` sends `<config name>.json` (e.g. `pilottone_offline.json`).
  - Each scan's log starts with `Config (<where it came from>)`, and its saved `settings` record it as
    `configSource`.

## Usage

### Capturing data from the scanner

You can capture FIRE's raw MRD stream from the scanner using
```
nc -l 9002 > fire-raw-mrd.dat
```
- Change `9002` to match the port in `wip_070_fire_PilotTone.ini`

You can capture data in ismrmd h5 format from the scanner using:
```
python main.py -s -S /path/to/saved_data
```

### Data Conversion

You can convert FIRE's raw MRD stream to ismrmd h5:
```
mrd-stream2h5 fire-raw-mrd.dat fire-raw-mrd.h5
```
(or `python stream2h5.py ...` from the repository)

You can convert a meas.dat from twix to h5 with
```
siemens_to_ismrmrd --measNum 2 -f meas_MID01104_FID99032_gre4mm_sag.dat -o meas_MID01104_FID99032_gre4mm_sag.h5
```
- You probably want `--measNum 2` to conver the second series (not the coil adjust)

## Playback

Away from the scanner, you can replay a raw MRD stream and send it to the server.

First start the server
```
python main.py -d pilottone
```

Then in another terminal:
```
nc -N localhost 9002 < fire-raw-mrd.dat > /dev/null
```
- Change `9002` to match the port in `wip_070_fire_PilotTone.ini`

You can replay ismrmd h5 data with
```
python client.py /path/to/h5_data
```
- Use `-c pilottone` to force it to use the pilot tone analysis
- Use `-o /tmp/ismrmrd-server-turd-bucket.h5` to keep all output in one place
  

