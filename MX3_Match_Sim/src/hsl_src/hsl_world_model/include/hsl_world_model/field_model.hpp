// field_model.hpp — HSL 2026 경기장 치수와 랜드마크 지도.
// 랜드마크 배치는 humanoid_localization/vision/field_landmarks.py 와 같다 (YOLO 클래스 체계도 같음).
// 좌표: 원점 = 센터, +x = 상대 골대 방향, +y = 왼쪽, 단위 m.
#pragma once
#include <Eigen/Dense>
#include <string>
#include <vector>

namespace hsl
{

enum class LandmarkType { GOALPOST = 1, X = 5, T = 6, L = 7 };

struct FieldModel
{
  double length = 9.0, width = 6.0;
  double center_circle_radius = 0.75;
  double penalty_mark_dist = 1.5;      // 골라인에서 페널티 마크까지
  double penalty_area_depth = 2.0, penalty_area_width = 4.0;
  double goal_area_depth = 1.0, goal_area_width = 3.0;
  double goal_width = 1.8;
  double border_x = 1.0, border_y = 1.0;  // Separate goal-line/touchline carpet margins.

  double hl() const { return length / 2; }
  double hw() const { return width / 2; }

  static FieldModel forProfile(const std::string & profile)
  {
    FieldModel f;
    if(profile == "practice") {        // User's drawing, 2026-11-12; not a scaled S-Field.
      f.length = 3.9; f.width = 2.6;
      f.center_circle_radius = 0.35; f.penalty_mark_dist = 0.65;
      f.penalty_area_depth = 0.87; f.penalty_area_width = 2.17;
      f.goal_area_depth = 0.43; f.goal_area_width = 1.30;
      f.goal_width = 1.13;
      f.border_x = 0.55; f.border_y = 0.20;  // Carpet is 5.0 x 3.0 m.
    }
    return f;   // 기본 s_field (9 x 6)
  }

  bool insideCarpet(const Eigen::Vector2d & p) const
  {
    return std::abs(p.x()) <= hl() + border_x && std::abs(p.y()) <= hw() + border_y;
  }

  struct Line { Eigen::Vector2d start, end; bool halfway = false; };

  /** 필드 직선 (센터서클 제외). 모두 x 축 또는 y 축에 평행하다 (B-Human 선 등록이 이 성질을 쓴다). */
  std::vector<Line> lines() const
  {
    std::vector<Line> out;
    auto seg = [&](double x0, double y0, double x1, double y1, bool hw_line = false) {
      out.push_back({Eigen::Vector2d(x0, y0), Eigen::Vector2d(x1, y1), hw_line});
    };
    const double pa = hl() - penalty_area_depth, ga = hl() - goal_area_depth;
    const double pw = penalty_area_width / 2, gw = goal_area_width / 2;
    seg(-hl(), hw(), hl(), hw());           // 터치라인
    seg(-hl(), -hw(), hl(), -hw());
    seg(0, -hw(), 0, hw(), true);           // 하프라인
    for(const double s : {-1.0, 1.0}) {
      seg(s * hl(), -hw(), s * hl(), hw()); // 골라인
      seg(s * pa, -pw, s * pa, pw);         // 페널티 에어리어 앞선
      seg(s * pa, pw, s * hl(), pw);        //            옆선
      seg(s * pa, -pw, s * hl(), -pw);
      seg(s * ga, -gw, s * ga, gw);         // 골 에어리어 앞선
      seg(s * ga, gw, s * hl(), gw);        //          옆선
      seg(s * ga, -gw, s * hl(), -gw);
    }
    return out;
  }

  std::vector<Eigen::Vector2d> landmarks(LandmarkType t) const
  {
    std::vector<Eigen::Vector2d> out;
    auto quad = [&](double x, double y) {
      out.emplace_back(x, y); out.emplace_back(x, -y); out.emplace_back(-x, y); out.emplace_back(-x, -y);
    };
    const double pxi = hl() - penalty_area_depth, gxi = hl() - goal_area_depth;
    switch(t) {
      case LandmarkType::GOALPOST: quad(hl(), goal_width / 2); break;
      case LandmarkType::X:
        out.emplace_back(0, 0);
        out.emplace_back(hl() - penalty_mark_dist, 0);
        out.emplace_back(-(hl() - penalty_mark_dist), 0);
        out.emplace_back(0, center_circle_radius);
        out.emplace_back(0, -center_circle_radius);
        break;
      case LandmarkType::T:
        quad(hl(), goal_area_width / 2);
        quad(hl(), penalty_area_width / 2);
        out.emplace_back(0, hw()); out.emplace_back(0, -hw());
        break;
      case LandmarkType::L:
        quad(hl(), hw());
        quad(pxi, penalty_area_width / 2);
        quad(gxi, goal_area_width / 2);
        break;
    }
    return out;
  }
};

}  // namespace hsl
