import cv2
import numpy as np
import os

# Segmentasyon tabanlı ROI'li yöntem

# Tespit ve panel yapılandırma sabitleri
MIN_MATCH_COUNT = 5             # Bir tespit için gereken minimum iyi eşleşme sayısı
FRAME_SKIP = 3                  # İşleme hızını artırmak için her n. karede bir işleme
PANEL_CELL_WIDTH = 150          # Paneldeki her hazmat hücresi için sabit genişlik
PANEL_CELL_HEIGHT = 150         # Paneldeki her hazmat hücresi için sabit yükseklik
GRID_COLS = 3                   # 3 sütun (önceden 5 sütun idi)
GRID_ROWS = 5                   # 5 satır (önceden 3 satır idi)

# Genel eşik değerleri
VANISH_THRESHOLD = 150          # Bir obje bu kadar kare boyunca görülmezse kaybolmuş sayılır
RECOGNITION_THRESHOLD = 200     # Centroid aralığı bu eşikten küçükse aynı obje kabul edilir

def load_templates(template_folder, sift):
    """
    Belirtilen klasörden template (şablon) resimlerini yükler ve SIFT özelliklerini hesaplar.
    """
    templates = {}
    features = {}
    for filename in os.listdir(template_folder):
        if filename.lower().endswith('.png'):
            path = os.path.join(template_folder, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Template {filename} yüklenemedi.")
                continue
            obj_name = os.path.splitext(filename)[0]
            templates[obj_name] = img
            kp, desc = sift.detectAndCompute(img, None)
            features[obj_name] = (kp, desc)
    return templates, features

def compute_sift_features(image, sift):
    """
    Verilen görüntü için SIFT özelliklerini (anahtar noktalar ve descriptor'lar) hesaplar.
    """
    image = cv2.convertScaleAbs(image)
    image = np.ascontiguousarray(image)
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    image = np.array(image, dtype=np.uint8)
    if image.size == 0:
        return [], None
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(desc1, desc2):
    """
    İki görüntü arasındaki SIFT descriptor'larını, Lowe'nun oran testi kullanarak eşleştirir.
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)
    good_matches = []
    for match in matches:
        if len(match) < 2:
            continue
        m, n = match
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)
    return good_matches

def detect_barrels(frame):
    """
    Giriş karedeki kırmızı ve mavi varilleri, HSV renk eşikleme yöntemiyle tespit eder.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0, 70, 50]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 70, 50]), np.array([180, 255, 255])
    lower_blue, upper_blue = np.array([100, 150, 0]), np.array([140, 255, 255])
    
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for cnt in red_contours:
        if cv2.contourArea(cnt) > 6000:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append(("Kirmizi Varil", (x, y, w, h)))
    for cnt in blue_contours:
        if cv2.contourArea(cnt) > 15000:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append(("Mavi Varil", (x, y, w, h)))
    return detections

def detect_hazmat_signs_with_confidence(frame, templates, features, sift):
    """
    Giriş karedeki merkezi bölgede hazmat template'lerini SIFT eşleştirmeleriyle tespit eder.
    Her template için eşleşme sayısına göre % güven hesaplanır.
    """
    confidences = {}
    detections = {}
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scene_kp, scene_desc = compute_sift_features(frame_gray, sift)
    if scene_desc is None:
        for obj_name in templates:
            confidences[obj_name] = 0
            detections[obj_name] = None
        return confidences, detections

    for obj_name, (tpl_kp, tpl_desc) in features.items():
        if tpl_desc is None:
            confidences[obj_name] = 0
            detections[obj_name] = None
            continue
        good_matches = match_features(tpl_desc, scene_desc)
        conf = min(100, int(100 * (len(good_matches) / MIN_MATCH_COUNT)))
        if len(good_matches) >= MIN_MATCH_COUNT:
            src_pts = np.float32([tpl_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([scene_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 220.0)
            if M is not None:
                h, w = templates[obj_name].shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                area = cv2.contourArea(dst)
                x_bb, y_bb, w_bb, h_bb = cv2.boundingRect(dst)
                rect_area = w_bb * h_bb
                if rect_area > 2000:
                    ratio = area / rect_area
                    if (0.4 <= ratio <= 0.55) and area < 30000:
                        detections[obj_name] = dst
                        confidences[obj_name] = conf
                        continue
        detections[obj_name] = None
        confidences[obj_name] = conf
    return confidences, detections

def get_diamond_points(bbox):
    """
    Verilen bbox (x, y, w, h) için elmas şeklinde (diamond) dört nokta hesaplar.
    """
    x, y, w, h = bbox
    pts = np.array([
        [int(x + w/2), y],
        [x+w, int(y + h/2)],
        [int(x + w/2), y+h],
        [x, int(y + h/2)]
    ], np.int32)
    pts = pts.reshape((-1, 1, 2))
    return pts

def segment_hazmat_roi(frame, bbox):
    """
    Verilen bbox (hazmat ROI) üzerinden segmentasyon uygulayarak daha iyi bir kontur elde eder.
    Basit Otsu eşikleme ve morfolojik işlemlerle çalışır.
    """
    x, y, w, h = bbox
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:  # Added: check for empty ROI
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        # Kontur noktalarını orijinal koordinatlara offsetleriz
        largest = largest + np.array([[x, y]])
        return largest
    else:
        return None

def update_hazmat_panel(templates, confidences, detections, counts):
    """
    Hazmat şablonlarını 3x5 ızgara şeklinde gösteren yan paneli oluşturur.
    Her hücre sabit boyuta yeniden boyutlanır; tespit yoksa soluk, varsa normal parlaklıkta gösterilir.
    Alt satıra, toplam tespit sayısı ("Count: X") yazdırılır.
    """
    cell_images = []
    sorted_keys = sorted(templates.keys())
    for obj_name in sorted_keys:
        tpl_img = cv2.cvtColor(templates[obj_name], cv2.COLOR_GRAY2BGR)
        img_resized = cv2.resize(tpl_img, (PANEL_CELL_WIDTH, PANEL_CELL_HEIGHT))
        if detections.get(obj_name) is None:
            display_img = cv2.convertScaleAbs(img_resized, alpha=0.3, beta=0)
        else:
            display_img = img_resized.copy()
        # İşaret ismi, confidence ve total count bilgilerini ekleyelim
        cv2.putText(display_img, obj_name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(display_img, f"{confidences.get(obj_name, 0)}%", (5, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(display_img, f"Count: {counts.get(obj_name, 0)}", (5, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cell_images.append(display_img)
    
    rows = []
    num_cells = GRID_COLS * GRID_ROWS
    for r in range(0, len(cell_images), GRID_COLS):
        row_cells = cell_images[r:r + GRID_COLS]
        if len(row_cells) < GRID_COLS:
            # Add empty cells if the row is incomplete
            empty_cell = np.zeros((PANEL_CELL_HEIGHT, PANEL_CELL_WIDTH, 3), dtype=np.uint8)
            row_cells.extend([empty_cell] * (GRID_COLS - len(row_cells)))
        row = np.hstack(row_cells)
        rows.append(row)
    panel = np.vstack(rows) if rows else None
    return panel

def show_pause(frame, templates, hazmat_confidences, hazmat_detections, hazmat_counts):
    """
    Yardımcı fonksiyon: Video çerçevesi ve hazmat panelini birleştirir,
    tek pencereye yerleştirir ve waitKey(0) ile duraklatır.
    """
    panel_img = update_hazmat_panel(templates, hazmat_confidences, hazmat_detections, hazmat_counts)
    if panel_img is not None:
        scale = frame.shape[0] / panel_img.shape[0]
        new_width = int(panel_img.shape[1] * scale)
        panel_resized = cv2.resize(panel_img, (new_width, frame.shape[0]))
        combined_frame = np.hstack((frame, panel_resized))
    else:
        combined_frame = frame
    cv2.imshow("Video and Hazmat Panel", combined_frame)
    cv2.waitKey(0)

def process_video(video_path, templates, features, sift):
    """
    Videoyu kare kare işler:
      - Merkezi bölgede hazmat tespiti yapılır.
      - Variller centroid bazlı tanınır.
      - Hazmat işareti için, tüm şablonlar arasında en yüksek güvenli tespit seçilir
        ve centroid izleme mantığıyla takip edilir.
      - Tespit yapıldığında, birleşik pencere içerisinde duraklatma sağlanır.
      - Ekranda yalnızca o karede görünür objeler çizilir.
      - Ek olarak, hazmat işareti için baklava (diamond) sınırlar çizilip,
        segmentasyon tabanlı ROI ile tespit alanı iyileştirilir.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Video açılmadı.")
        return

    reported_hazmat = {key: False for key in templates.keys()}
    tracked_barrels = {}    # Barrel tracking için: ID → detaylar
    tracked_hazmat = None   # Hazmat için tek obje (en yüksek güven tespiti)
    next_barrel_id = 1
    hazmat_counts = {key: 0 for key in templates.keys()}

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        if frame_index % FRAME_SKIP != 0:
            continue

        # Her kare başında izlenen objelerin görünürlüğünü sıfırla
        for track in tracked_barrels.values():
            track['visible'] = False
        if tracked_hazmat is not None:
            tracked_hazmat['visible'] = False

        # Merkezi bölge (hazmat tespiti için)
        h_frame, w_frame = frame.shape[:2]
        x_start = w_frame // 3
        y_start = h_frame // 3
        x_end = 2 * w_frame // 3
        y_end = 2 * h_frame // 3
        proc_frame = frame[y_start:y_end, x_start:x_end]

        # Hazmat tespitleri: Tüm şablonlar için confidence ve poligon hesaplanır
        hazmat_confidences, hazmat_detections = detect_hazmat_signs_with_confidence(proc_frame, templates, features, sift)
        # En yüksek güvenli tespiti seç (varsa)
        selected_conf = 0
        selected_template = None
        selected_detection = None
        for temp_name, conf in hazmat_confidences.items():
            if hazmat_detections[temp_name] is not None and conf > selected_conf:
                selected_conf = conf
                selected_template = temp_name
                selected_detection = hazmat_detections[temp_name]
        
        # Hazmat izleme: Tespit var ise güncelle, yoksa görünmez yap
        if selected_detection is not None:
            pts = (selected_detection + np.array([[[x_start, y_start]]])).astype(int)
            x_h, y_h, w_h, h_h = cv2.boundingRect(pts)
            haz_centroid = (x_h + w_h/2, y_h + h_h/2)
            # Segmentation based ROI: İsteğe bağlı kontur iyileştirmesi
            seg_contour = segment_hazmat_roi(frame, (x_h, y_h, w_h, h_h))
            if seg_contour is not None:
                # Eğer segmentasyon başarı verirse, bbox'ı yeniden hesapla
                x_seg, y_seg, w_seg, h_seg = cv2.boundingRect(seg_contour)
                # İsterseniz seg_contour direkt kullanılabilir de; burada bbox olarak güncelliyoruz.
                x_h, y_h, w_h, h_h = x_seg, y_seg, w_seg, h_seg

            if tracked_hazmat is not None:
                dist = np.linalg.norm(np.array(tracked_hazmat['centroid']) - np.array(haz_centroid))
                if dist < RECOGNITION_THRESHOLD:
                    tracked_hazmat['centroid'] = haz_centroid
                    tracked_hazmat['bbox'] = (x_h, y_h, w_h, h_h)
                    tracked_hazmat['last_seen'] = frame_index
                    tracked_hazmat['visible'] = True
                else:
                    tracked_hazmat = {
                        'template': selected_template,
                        'centroid': haz_centroid,
                        'bbox': (x_h, y_h, w_h, h_h),
                        'first_seen': frame_index,
                        'last_seen': frame_index,
                        'visible': True
                    }
                    hazmat_counts[selected_template] = hazmat_counts.get(selected_template, 0) + 1
                    print(f"Hazmat '{selected_template}' tespit edildi, {frame_index}. kare")
                    cv2.putText(frame, f"Hazmat {selected_template} tespit edildi.",
                                (x_h, y_h+h_h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                    show_pause(frame, templates, hazmat_confidences, hazmat_detections, hazmat_counts)
            else:
                tracked_hazmat = {
                    'template': selected_template,
                    'centroid': haz_centroid,
                    'bbox': (x_h, y_h, w_h, h_h),
                    'first_seen': frame_index,
                    'last_seen': frame_index,
                    'visible': True
                }
                hazmat_counts[selected_template] = hazmat_counts.get(selected_template, 0) + 1
                print(f"Hazmat '{selected_template}' tespit edildi, {frame_index}. kare")
                cv2.putText(frame, f"Hazmat {selected_template} tespit edildi.",
                            (x_h, y_h+h_h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                show_pause(frame, templates, hazmat_confidences, hazmat_detections, hazmat_counts)
        else:
            if tracked_hazmat is not None:
                tracked_hazmat['visible'] = False
                if (frame_index - tracked_hazmat['last_seen'] > VANISH_THRESHOLD):
                    print(f"Hazmat '{tracked_hazmat['template']}' {frame_index}. karede kayboldu")
                    tracked_hazmat = None

        # Barrel tanıma: Centroid bazlı takip (varilleri güncelle ve görünür yap)
        barrel_detections = detect_barrels(frame)
        for (barrel_type, (x, y, w, h)) in barrel_detections:
            centroid = (x + w/2, y + h/2)
            matched_track_id = None
            min_distance = RECOGNITION_THRESHOLD
            for track_id, track in tracked_barrels.items():
                if track['type'] == barrel_type:
                    dist = np.linalg.norm(np.array(track['centroid']) - np.array(centroid))
                    if dist < min_distance:
                        min_distance = dist
                        matched_track_id = track_id
            if matched_track_id is not None:
                tracked_barrels[matched_track_id]['centroid'] = centroid
                tracked_barrels[matched_track_id]['bbox'] = (x, y, w, h)
                tracked_barrels[matched_track_id]['last_seen'] = frame_index
                tracked_barrels[matched_track_id]['visible'] = True
            else:
                track_id = next_barrel_id
                next_barrel_id += 1
                tracked_barrels[track_id] = {
                    'id': track_id,
                    'type': barrel_type,
                    'centroid': centroid,
                    'bbox': (x, y, w, h),
                    'first_seen': frame_index,
                    'last_seen': frame_index,
                    'visible': True
                }
                print(f"{barrel_type} varil {track_id}, {frame_index}. kare'de görüldü")
                cv2.putText(frame, f"{barrel_type} {track_id} tespit edildi.",
                            (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                show_pause(frame, templates, hazmat_confidences, hazmat_detections, hazmat_counts)

        # Varil vanish kontrolü: Güncellenmeyenleri sil
        for track_id in list(tracked_barrels.keys()):
            if frame_index - tracked_barrels[track_id]['last_seen'] > VANISH_THRESHOLD:
                print(f"{tracked_barrels[track_id]['type']} {track_id}, {frame_index}. karede kayboldu")
                del tracked_barrels[track_id]

        # Çizim: Hazmat işareti için hem baklava (diamond) sınırlar hem de bounding box çizilsin.
        if tracked_hazmat is not None and tracked_hazmat.get('visible', False):
            (x_h, y_h, w_h, h_h) = tracked_hazmat['bbox']
            # Draw bounding box (rectangle)
            cv2.rectangle(frame, (x_h, y_h), (x_h+w_h, y_h+h_h), (0, 255, 255), 2)  # Yellow rectangle
            # Draw diamond bounders (baklava)
            diamond_pts = get_diamond_points((x_h, y_h, w_h, h_h))
            cv2.polylines(frame, [diamond_pts], isClosed=True, color=(0, 255, 0), thickness=3)
            cv2.putText(frame, f"Hazmat {tracked_hazmat['template']}",
                        (x_h, y_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Çizim: Sadece görünür olan varillerin kutuları çizilsin
        for track_id, track in tracked_barrels.items():
            if track.get('visible', False):
                (x, y, w, h) = track['bbox']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"{track['type']} {track_id}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # Her döngüde birleşik pencereyi güncelleyelim
        panel_img = update_hazmat_panel(templates, hazmat_confidences, hazmat_detections, hazmat_counts)
        if panel_img is not None:
            scale = frame.shape[0] / panel_img.shape[0]
            new_width = int(panel_img.shape[1] * scale)
            panel_resized = cv2.resize(panel_img, (new_width, frame.shape[0]))
            combined_frame = np.hstack((frame, panel_resized))
        else:
            combined_frame = frame

        cv2.imshow("Video and Hazmat Panel", combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    """
    Ana fonksiyon:
      - SIFT dedektörünü başlatır.
      - Hazmat template'lerini yükler.
      - Video işleyicisini çağırır.
    """
    sift = cv2.SIFT_create()
    template_folder = os.path.join(os.path.dirname(__file__), "hazmats", "hazmats")
    video_path = os.path.join(os.path.dirname(__file__), "video.mp4")
    
    templates, features = load_templates(template_folder, sift)
    if not templates:
        print("Hiç template yüklenemedi.")
        return

    process_video(video_path, templates, features, sift)

if __name__ == "__main__":
    main()
