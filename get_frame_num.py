import os

seq_name = ['a06845dd7d1d808e4f4743b7f08f2bf75a9a72264d4fb16505caf6e334611003',
            '57cb54c2cde2789359ecf11b9b9b8207c6a79b7aa27f15a69d7e9a1c2caad912',
            'fae057c83b04868424da3bb7139e29b3f328d5a93aaa9e617e825b93422d92c5',
            'af31d741db80475c531bb7182ad0536df9dc88a6876fa38386dd5db850d86051',
            'd0a99fb6b64e60d7754265586481ec43968e8fd97e7e4437332bb182d7548cb3',
            '97d6ac9d81b64bf909bf4898072bb20492522ae182918e763a86b56745890add',
            'd73059fe0ed42169f7e98ff7401d00479a7969753eb80af9846176a42543ccb0',
            '23e266612abe7b8767587d6e77a5eb3c6b8a71c6bf4c4ff2b1c11cc478cc7244',
            '9a6379abea3fc820ca60afb9a60092d41b3a772ff348cfec92c062f6187f85e2',
            '7c7d58e4f82772f627d5cbe3df3b08573d5bd7a58639387b865449d5a550bbda',
            '29aabdd9d3065802c21e2d828561c205d563e79d39d1e10a18f961b5b5bf0cad',
            '7b0eaacc48c9b5ea0edf5dcf352d913fd0cf3f79ae149e94ada89ba1e772e711',
            '0442d8bdf9902226bfb38fbe039840d4f8ebe5270eda39d7dba56c2c3ae5becc',
            'b7ee0264612a6ca6bf2bfa03df68acf4af9bb5cac34f7ad43fe30fa4b7bc4824',
            '8db183688ce3e59461355e2c7cc97b3aee9f514a2e28260ead5a3ccf2000b079',
            '8cbafab285e74614f10d3a8bf9ee94434eacae6332f5f10fe1e50bfe5de9ec33',
            '318c694f5c83b78367da7e6584a95872510db8544f815120a86923aff00f5ff9',
            '04ca8d2ac3af26ad4c5b14cf214e0d7c317c953e804810829d41645fdce1ad88',
            '1e3224380c76fb4cad0a8d3a7c74a8d5bf0688d13df15f23acd2512de4374cb4',
            '04a1274a93ec6a36ad2c1cb5eb83c3bdf2cf05bbe01c70a8ca846a7f9fa4b550',
            '0d49152a92ce3b843968bf2e131ea5bc5e409ab056196e8c373f9bd2d31b303d',
            '5d8f03cf5c6a469004a0ca73948ad64fa6d222b3b807f155a66684387f5d208a',
            '0e1474478f33373566b4fbd6b357cf6b65015a6f4aa646754e065bf4a1b43c15',
            '0659b03fb82cae130fef6a931755bbaae6e7bd88f58873df1ae98d2145dba9ce',
            'a89f641b8dd2192f6f8b0ae75e3a24388b96023b21c63ff67bb359628f5df6de',
            '209921b14cef20d62002e2b0c21ad692226135b52fee7eead315039ca51c470c',
            '917d1b33f0e20d2d81471c3a0ff7adbef9e1fb7ee184b604880b280161ffdd56',
            '9ce4af9a3b304b4b5387f27bca137ce1f0f35c12837c753fc17ea9bb49eb8ec5',
            '393608bbbf2ac4d141ce6a3616a2364a2071539acb1969032012348c5817ef3c',
            '9299df423938da4fd7f51736070420d2bb39d33972729b46a16180d07262df12']

frame_num = []

src_pth = os.path.join(os.getenv('DATAROOT'), 'CLIC_2022/test_video_30/rgb')

for seq in seq_name:
    idx = 0
    while True:
        idx += 1
        pth = f'{src_pth}/{seq}/frame_{idx}.png'
        if not os.path.exists(pth):
            break
    with open('CLIC_frames.txt', 'a') as f:
        f.write(f'\'{seq}\': {idx-1}\n')
    frame_num.append(idx-1)
print(frame_num)