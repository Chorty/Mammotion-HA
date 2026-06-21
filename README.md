# Mammotion - Home Assistant Integration [![Discord](https://img.shields.io/discord/1247286396297678879)](https://discord.gg/vpZdWhJX8x)

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=mikey0000&repository=mammotion-HA&category=Integration)

💬 [Join us on Discord](https://discord.gg/vpZdWhJX8x)

This integration allows you to control and monitor Mammotion products, e.g robot lawn mowers using Home Assistant.

⚠️ **Please note:** This integration is still a work in progress. You may encounter unfinished features or bugs. If you come across any issues, please open an issue on the GitHub repository. 🐛

## Roadmap 🗺️

- [x] Bluetooth (BLE) support
- [x] Wi-Fi support (Including SIM 3G)
- [x] Run existing scheduled tasks
- [ ] Create and edit schedules
- [x] Map display and area selection
- [ ] Edit zone geometry and map metadata
- [x] Firmware updates (experimental)
- [x] Automations
- [ ] Expand tested device and firmware compatibility

## Features ✨

- Start, stop, pause, and dock the mower
- Monitor the mower's status (e.g., mowing, charging, idle)
- View the mower's battery level
- Start a mow based on configuration
- Start an existing scheduled task/s
- More features being added all the time!
- Render the mower's path as a map camera for use with front-end map cards
- Install available mower firmware updates (experimental)

### Connection modes

- **Wi-Fi/cloud:** Supports normal monitoring, control, maps, schedules, camera features on compatible models, and firmware checks. Use a secondary Mammotion account shared with the mower to avoid signing the mobile app out.
- **Bluetooth:** Used directly or as a fallback when a supported mower is reachable through Home Assistant Bluetooth. A Bluetooth proxy is recommended when Home Assistant is not physically near the mower.
- **Multiple accounts:** Separate config entries are supported. Services resolve the selected entity to the correct account and mower.

## Prerequisites 📋
> [!WARNING]
> **Home Assistant Minimum Version 2025.3.0**
- A second account with your mower/s shared to it for using Wi-Fi (If you use your primary accouunt it will log you out of your mobile app)
- (Optional)[Bluetooth proxy for Home Assistant](https://esphome.io/components/bluetooth_proxy.html)

## Installation 🛠️

This integration can be installed using [HACS](https://hacs.xyz/)

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-41BDF5.svg?style=for-the-badge)](https://github.com/hacs/integration)

This integration is not available in the default HACS store. You will need to add it as a custom repository.

1. Go to HACS > Integrations and click on the 3 dots in the top right corner.
2. Select "Custom repositories".
3. In the "Repository" field, paste this URL: `https://github.com/mikey0000/Mammotion-HA`
4. For "Category", select "Integration".
5. Click "Add".
6. You can now search for "Mammotion" within HACS and install it.
7. After installation, restart Home Assistant.
8. Go to **Settings > Devices & Services** and click **+ Add Integration** to configure Mammotion.

## Usage 🎮

### Getting Started
See the wiki for how to [get started](https://github.com/mikey0000/Mammotion-HA/wiki/Getting-Started)

Once the integration is set up, you can control and monitor your Mammotion mower using Home Assistant. 🎉

### Map display

This integration creates a `camera` entity that draws the mower's path and a companion `vacuum` entity for compatibility with map cards. When using the [Lovelace Xiaomi Vacuum Map card](https://github.com/PiotrMachowski/lovelace-xiaomi-vacuum-map-card), set the card's `entity` to the mower's vacuum entity and `camera_entity` to the map camera.

### Reauthentication and diagnostics

If Mammotion credentials expire, Home Assistant will prompt you to reauthenticate the existing integration entry. Use credentials for the same Mammotion account; devices and entity IDs are preserved.

Diagnostics are intentionally bounded and exclude credentials, account identifiers, serial numbers, MAC addresses, coordinates, map geometry, tokens, session data, and raw protocol payloads.

## Troubleshooting 🔧

If you encounter any issues with the Mammotion integration, please check the Home Assistant logs for error messages. You can also try the following troubleshooting steps:

- Verify that you have Bluetooth proxy setup with Home Assistant.
- Ensure that your mower is connected to your home network and accessible from Home Assistant.
- Restart Home Assistant and check if the issue persists.
- If the integration reports an authentication failure, open **Settings > Devices & services**, select Mammotion, and complete the reauthentication prompt.

### Known limitations

- Schedule creation and editing must currently be performed in the Mammotion app; Home Assistant can run existing schedules.
- Map rendering and area selection are supported, but editing zone geometry is not.
- Firmware installation is experimental and depends on model, current firmware, cloud availability, and Mammotion account permissions.
- Camera streaming is unavailable on some mower generations and uses Mammotion's Agora service rather than native Home Assistant WebRTC.

## Contributing to Translations

We use Crowdin to manage our translations. If you'd like to contribute:

1. Visit our [Crowdin project page](https://crowdin.com/project/mammotion-ha)
2. Select the language you'd like to translate to
3. Start translating!

Your contributions will be automatically submitted as pull requests to this repository.


## PyMammotion Library 📚

This integration uses the [PyMammotion library](https://github.com/mikey0000/PyMammotion) to communicate with Mammotion mowers. PyMammotion provides a Python API for controlling and monitoring Mammotion robot mowers via MQTT, Cloud, and Bluetooth.

If the problem continues, please file an issue on the GitHub repository for further assistance. 🙏

## Support me
<a href='https://ko-fi.com/DenimJackRabbit' target='_blank'><img height='46' style='border:0px;height:46px;' src='https://az743702.vo.msecnd.net/cdn/kofi3.png?v=0' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>
### Referral Links
[Buy a Mammotion Lawn mower (Amazon)](https://amzn.to/4cOLULU)
[Buy a Mammotion Lawn mower (Mammotion)](https://mammotion.com/?ref=denimjackrabbit)

## Credits 👥

[![Contributors](https://contrib.rocks/image?repo=mikey0000/Mammotion-HA)](https://github.com/mikey0000/Mammotion-HA/graphs/contributors)
